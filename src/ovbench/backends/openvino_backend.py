"""Backend OpenVINO (optimum-intel). Funciona com qualquer causal LM já em formato OpenVINO IR."""
from __future__ import annotations

import time
from pathlib import Path

from ovbench.backends.base import GenerationResult, LLMBackend


def _resolver(model_path: str) -> Path:
    caminho = Path(model_path).expanduser().resolve()
    if not caminho.is_dir():
        raise FileNotFoundError(
            f"Pasta do modelo não encontrada: {caminho}\n"
            "Passe uma pasta com o modelo já convertido para OpenVINO "
            "(use 'ovbench convert' para converter/quantizar um modelo do Hugging Face)."
        )
    return caminho


def tokenizer_counter(model_path: str, trust_remote_code: bool = False):
    """Função texto->nº de tokens usando só o tokenizer (não carrega o modelo)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        _resolver(model_path), trust_remote_code=trust_remote_code, local_files_only=True)
    return lambda texto: len(tok.encode(texto, add_special_tokens=False))


class OpenVINOBackend(LLMBackend):
    name = "openvino"

    def __init__(self, model_path: str, ov_config: dict | None = None, device: str = "CPU",
                 enable_thinking: bool | None = None, trust_remote_code: bool = False,
                 instance_id: int = 0):
        self.model_path = model_path
        self.ov_config = dict(ov_config or {})
        self.device = device
        self.enable_thinking = enable_thinking
        self.trust_remote_code = trust_remote_code
        self.instance_id = instance_id
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        caminho = _resolver(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            caminho, trust_remote_code=self.trust_remote_code, local_files_only=True)
        self.model = OVModelForCausalLM.from_pretrained(
            caminho, device=self.device, local_files_only=True, export=False,
            ov_config=self.ov_config, trust_remote_code=self.trust_remote_code)
        self.model.compile()
        # Geração gulosa/determinística: mede velocidade sem ruído de amostragem.
        gc = self.model.generation_config
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        gc.do_sample = False

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _renderizar(self, messages: list[dict]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            kwargs = {}
            if self.enable_thinking is not None:
                kwargs["enable_thinking"] = self.enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **kwargs)
        # Modelo base sem chat template: concatenação simples.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def generate(self, messages, max_new_tokens=256, min_new_tokens=None) -> GenerationResult:
        from transformers.generation.streamers import BaseStreamer

        class _Cronometro(BaseStreamer):
            """O 1º put() recebe o prompt; o 2º é o 1º token gerado."""
            def __init__(self):
                self.chamadas = 0
                self.t_primeiro = None

            def put(self, value):
                self.chamadas += 1
                if self.chamadas == 2 and self.t_primeiro is None:
                    self.t_primeiro = time.perf_counter()

            def end(self):
                pass

        prompt = self._renderizar(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        n_in = int(inputs["input_ids"].shape[1])
        cron = _Cronometro()
        extra = {}
        if min_new_tokens:
            extra["min_new_tokens"] = min_new_tokens
        t0 = time.perf_counter()
        saida = self.model.generate(
            **inputs, streamer=cron, use_cache=True, do_sample=False,
            max_new_tokens=max_new_tokens, return_dict_in_generate=True, **extra)
        total = time.perf_counter() - t0
        gerados = saida.sequences[0][n_in:]
        texto = self.tokenizer.decode(gerados, skip_special_tokens=True)
        ttft = None if cron.t_primeiro is None else cron.t_primeiro - t0
        return GenerationResult(texto, n_in, len(gerados), ttft, total,
                                hit_limit=len(gerados) >= max_new_tokens)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
