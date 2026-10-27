"""Estatísticas e medição de memória."""
from __future__ import annotations

import math
import statistics
import threading
from dataclasses import dataclass, asdict


def percentile(valores: list[float], q: float) -> float:
    if not valores:
        return float("nan")
    s = sorted(valores)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(valores: list[float]) -> dict:
    if not valores:
        return {"n": 0}
    return {
        "n": len(valores),
        "mean": statistics.fmean(valores),
        "std": statistics.stdev(valores) if len(valores) > 1 else 0.0,
        "min": min(valores),
        "p50": percentile(valores, 50),
        "p95": percentile(valores, 95),
        "max": max(valores),
    }


@dataclass
class Sample:
    prompt_id: str
    repetition: int
    n_input_tokens: int
    n_output_tokens: int
    total_s: float            # só inferência
    ttft_s: float | None
    wait_s: float = 0.0       # espera por instância livre (modo pool)
    valid: bool | None = None
    hit_limit: bool = False

    @property
    def e2e_s(self) -> float:
        return self.wait_s + self.total_s

    @property
    def decode_tps(self) -> float | None:
        """Tokens/s só na fase de decodificação (exclui o prefill)."""
        if self.ttft_s is None or self.n_output_tokens < 2 or self.total_s <= self.ttft_s:
            return None
        return (self.n_output_tokens - 1) / (self.total_s - self.ttft_s)

    @property
    def overall_tps(self) -> float | None:
        return self.n_output_tokens / self.total_s if self.total_s > 0 else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(e2e_s=self.e2e_s, decode_tps=self.decode_tps, overall_tps=self.overall_tps)
        return d


def aggregate(samples: list[Sample]) -> dict:
    def col(f):
        return [v for v in (f(s) for s in samples) if v is not None]

    validados = [s.valid for s in samples if s.valid is not None]
    return {
        "latencia_ms": summarize([s.total_s * 1000 for s in samples]),
        "e2e_ms": summarize([s.e2e_s * 1000 for s in samples]),
        "espera_ms": summarize([s.wait_s * 1000 for s in samples]),
        "ttft_ms": summarize([v * 1000 for v in col(lambda s: s.ttft_s)]),
        "decode_tok_s": summarize(col(lambda s: s.decode_tps)),
        "tok_s_geral": summarize(col(lambda s: s.overall_tps)),
        "tokens_entrada": summarize([float(s.n_input_tokens) for s in samples]),
        "tokens_saida": summarize([float(s.n_output_tokens) for s in samples]),
        "formato_valido": (sum(validados) / len(validados)) if validados else None,
        "atingiu_limite": sum(s.hit_limit for s in samples) / len(samples) if samples else None,
    }


class PeakMemory:
    """Amostra o RSS do processo em thread; `.peak_mb` fica None sem psutil."""

    def __init__(self, intervalo: float = 0.1):
        self.intervalo = intervalo
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self, proc):
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / 2**20
                self.peak_mb = rss if self.peak_mb is None else max(self.peak_mb, rss)
            except Exception:
                return
            self._stop.wait(self.intervalo)

    def __enter__(self):
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            return self
        self._thread = threading.Thread(target=self._loop, args=(proc,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
