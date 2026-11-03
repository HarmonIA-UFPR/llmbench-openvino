"""Execução do benchmark: modo instância (sequencial) e modo pool (concorrente)."""
from __future__ import annotations

import json
import queue
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Callable

from ovbench.metrics import PeakMemory, Sample, aggregate
from ovbench.workloads import Prompt

Factory = Callable[[dict, int], "LLMBackend"]  # noqa: F821


@dataclass
class RunConfig:
    repetitions: int = 3
    warmup: int = 1
    max_new_tokens: int = 256
    fixed_length: int | None = None  # força exatamente N tokens (compara modelos sem viés de tamanho)

    def gen_kwargs(self) -> dict:
        if self.fixed_length:
            return {"max_new_tokens": self.fixed_length, "min_new_tokens": self.fixed_length}
        return {"max_new_tokens": self.max_new_tokens}


def _gerar(backend, prompt: Prompt, rep: int, cfg: RunConfig, validator, saida: IO | None,
           rotulo: str, wait_s: float = 0.0) -> Sample:
    r = backend.generate(prompt.messages, **cfg.gen_kwargs())
    valido = validator(r.text) if validator else None
    s = Sample(prompt.id, rep, r.n_input_tokens, r.n_output_tokens, r.total_s, r.ttft_s,
               wait_s=wait_s, valid=valido,
               hit_limit=r.hit_limit and not cfg.fixed_length)  # com --fixed-length bater no limite é o esperado
    if saida is not None:
        saida.write(json.dumps({"config": rotulo, "prompt_id": prompt.id, "repeticao": rep,
                                "resposta": r.text, **s.to_dict()}, ensure_ascii=False) + "\n")
        saida.flush()
    return s


def _aquecer(backend, prompts: list[Prompt], n: int) -> None:
    for _ in range(n):
        backend.generate(prompts[0].messages, max_new_tokens=16)


def run_instance(factory: Factory, nome: str, ov_config: dict, prompts: list[Prompt], cfg: RunConfig,
                 validator=None, saida: IO | None = None, progress=print) -> dict:
    """Carrega UMA instância com `ov_config` e roda todos os prompts sequencialmente."""
    resultado = {"modo": "instancia", "config": nome, "ov_config": ov_config, "status": "ok"}
    backend = None
    try:
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            backend = factory(ov_config, 0)
            backend.load()
            resultado["load_s"] = time.perf_counter() - t0
            _aquecer(backend, prompts, cfg.warmup)
            amostras = []
            for rep in range(cfg.repetitions):
                for p in prompts:
                    amostras.append(_gerar(backend, p, rep, cfg, validator, saida, nome))
        resultado["pico_rss_mb"] = mem.peak_mb
        resultado["resumo"] = aggregate(amostras)
        resultado["por_prompt"] = {
            p.id: aggregate([s for s in amostras if s.prompt_id == p.id]) for p in prompts}
        resultado["amostras"] = [s.to_dict() for s in amostras]
    except Exception as e:  # uma config inválida não derruba as demais
        resultado.update(status="erro", erro=f"{type(e).__name__}: {e}")
        progress(f"  [erro] {nome}: {resultado['erro']}")
    finally:
        if backend is not None:
            backend.close()
    return resultado


def ordem_balanceada(n: int, rodada: int) -> list[int]:
    """Ordem dos n presets na `rodada`, em desenho Williams (quadrado latino balanceado).

    Com n par e `rodada` percorrendo n rodadas, cada preset ocupa cada posição exatamente uma vez e
    cada par ordenado "A logo depois de B" aparece exatamente uma vez -- neutraliza o efeito da posição
    no bloco e do preset anterior. Com n ímpar só a posição fica balanceada (rotação simples).
    """
    if n % 2 == 0:
        seq, a, b = [0], 1, n - 1
        while len(seq) < n:
            seq.append(a)
            a += 1
            if len(seq) < n:
                seq.append(b)
                b -= 1
    else:
        seq = list(range(n))
    return [(x + rodada) % n for x in seq]


def run_interleaved(factory: Factory, presets: dict[str, dict], prompts: list[Prompt], cfg: RunConfig,
                    validator=None, saida: IO | None = None, cooldown: float = 0.0, seed: int = 42,
                    ordem_modo: str = "balanceada", progress=print) -> list[dict]:
    """Intercala as repetições entre presets (A B C, B C A, ...).

    `ordem_modo='balanceada'` (padrão): rotaciona uma ordem-base (embaralhada pela semente) em desenho Williams,
    de modo que nenhum preset fique sempre cedo ou sempre tarde. Para o balanço completo use um nº de
    repetições múltiplo do nº de presets. `'aleatoria'`: embaralha a cada rodada (pode viciar posições).

    Throttling térmico e outras derivas lentas passam a afetar todos os presets por igual, em vez de
    penalizar quem roda por último. Custo: todas as instâncias ficam carregadas ao mesmo tempo
    (RAM ~ soma dos modelos) e `pico_rss_mb` é o pico do conjunto, igual para todos.
    """
    if ordem_modo not in ("balanceada", "aleatoria"):
        raise ValueError(f"ordem_modo inválido: {ordem_modo!r}")
    rng = random.Random(seed)
    resultados = {n: {"modo": "instancia", "config": n, "ov_config": c, "status": "ok", "intercalado": True}
                  for n, c in presets.items()}
    backends: dict[str, object] = {}
    amostras: dict[str, list[Sample]] = {n: [] for n in presets}
    ordem: list[list[str]] = []
    try:
        with PeakMemory() as mem:
            for n, c in presets.items():
                try:
                    t0 = time.perf_counter()
                    b = factory(c, 0)
                    b.load()
                    resultados[n]["load_s"] = time.perf_counter() - t0
                    _aquecer(b, prompts, cfg.warmup)
                    backends[n] = b
                except Exception as e:
                    resultados[n].update(status="erro", erro=f"{type(e).__name__}: {e}")
                    progress(f"  [erro] {n}: {resultados[n]['erro']}")
            ativos = list(backends)
            base = ativos[:]
            rng.shuffle(base)
            if ordem_modo == "balanceada" and ativos and cfg.repetitions % len(ativos):
                progress(f"  [aviso] {cfg.repetitions} repetições não é múltiplo de {len(ativos)} presets: "
                         "as posições ficam só aproximadamente balanceadas.")
            for rep in range(cfg.repetitions):
                if ordem_modo == "balanceada":
                    rodada = [base[i] for i in ordem_balanceada(len(base), rep)]
                else:
                    rodada = ativos[:]
                    rng.shuffle(rodada)
                ordem.append(rodada)
                progress(f"  rodada {rep + 1}/{cfg.repetitions}: {' > '.join(rodada)}")
                for i, n in enumerate(rodada):
                    if cooldown and (i or rep):
                        time.sleep(cooldown)
                    for pr in prompts:
                        amostras[n].append(_gerar(backends[n], pr, rep, cfg, validator, saida, n))
        for n in ativos:
            r = resultados[n]
            r["pico_rss_mb"] = mem.peak_mb
            r["resumo"] = aggregate(amostras[n])
            r["por_prompt"] = {pr.id: aggregate([s for s in amostras[n] if s.prompt_id == pr.id])
                               for pr in prompts}
            r["amostras"] = [s.to_dict() for s in amostras[n]]
            r["ordem_por_rodada"] = ordem
    finally:
        for b in backends.values():
            b.close()
    return list(resultados.values())


def run_pool(factory: Factory, nome: str, ov_configs: list[dict], prompts: list[Prompt], cfg: RunConfig,
             concorrencia: int | None = None, validator=None, saida: IO | None = None,
             progress=print) -> dict:
    """N instâncias carregadas de uma vez; requisições concorrentes disputam as instâncias livres."""
    n = len(ov_configs)
    concorrencia = concorrencia or n
    resultado = {"modo": "pool", "config": nome, "ov_configs": ov_configs, "instancias": n,
                 "concorrencia": concorrencia, "status": "ok"}
    backends = []
    try:
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            for i, c in enumerate(ov_configs):
                b = factory(c, i)
                b.load()
                backends.append(b)
            resultado["load_s"] = time.perf_counter() - t0
            for b in backends:
                _aquecer(b, prompts, cfg.warmup)
            livres: queue.Queue = queue.Queue()
            for b in backends:
                livres.put(b)
            tarefas = [(rep, p) for rep in range(cfg.repetitions) for p in prompts]

            def uma(tarefa):
                rep, p = tarefa
                t_ini = time.perf_counter()
                b = livres.get()  # bloqueia na fila; sem timeout: o que se mede é a espera
                espera = time.perf_counter() - t_ini
                try:
                    return _gerar(b, p, rep, cfg, validator, saida, nome, wait_s=espera)
                finally:
                    livres.put(b)

            t_total = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concorrencia) as ex:
                amostras = list(ex.map(uma, tarefas))
            duracao = time.perf_counter() - t_total
        resultado["pico_rss_mb"] = mem.peak_mb
        resultado["duracao_s"] = duracao
        resultado["requisicoes"] = len(amostras)
        resultado["throughput_req_s"] = len(amostras) / duracao
        resultado["tokens_saida_s"] = sum(s.n_output_tokens for s in amostras) / duracao
        resultado["resumo"] = aggregate(amostras)
        resultado["amostras"] = [s.to_dict() for s in amostras]
    except Exception as e:
        resultado.update(status="erro", erro=f"{type(e).__name__}: {e}")
        progress(f"  [erro] {nome}: {resultado['erro']}")
    finally:
        for b in backends:
            b.close()
    return resultado
