"""Backend falso (sem modelo): serve para testes e para validar o fluxo do CLI."""
from __future__ import annotations

import time

from ovbench.backends.base import GenerationResult, LLMBackend


class FakeBackend(LLMBackend):
    name = "fake"

    def __init__(self, ov_config: dict | None = None, instance_id: int = 0,
                 token_delay_s: float = 0.0005, prefill_s: float = 0.002, answer: str = '{"ok": true}'):
        self.ov_config = ov_config or {}
        self.instance_id = instance_id
        self.token_delay_s = token_delay_s
        self.prefill_s = prefill_s
        self.answer = answer
        self.loaded = False
        if self.ov_config.get("FAIL"):
            self._fail = True
        else:
            self._fail = False

    def load(self) -> None:
        if self._fail:
            raise RuntimeError("config inválida (simulada)")
        self.loaded = True

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages, max_new_tokens=256, min_new_tokens=None) -> GenerationResult:
        n_in = sum(self.count_tokens(m["content"]) for m in messages)
        n_out = min_new_tokens if min_new_tokens else max(2, min(max_new_tokens, 20))
        t0 = time.perf_counter()
        time.sleep(self.prefill_s)
        ttft = time.perf_counter() - t0
        time.sleep(self.token_delay_s * (n_out - 1))
        total = time.perf_counter() - t0
        return GenerationResult(self.answer, n_in, n_out, ttft, total, hit_limit=n_out >= max_new_tokens)
