"""Seleção de backend. Só OpenVINO (+ 'fake' para testes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ovbench.backends.base import GenerationResult, LLMBackend

__all__ = ["BackendSpec", "GenerationResult", "LLMBackend", "make_spec"]


@dataclass
class BackendSpec:
    name: str
    device: str
    factory: Callable[[dict, int], LLMBackend]  # (ov_config, instance_id) -> backend
    counter: Callable[[], Callable[[str], int]]  # devolve função texto -> nº de tokens


def make_spec(name: str, model: str | None, device: str = "CPU",
              enable_thinking: bool | None = None, trust_remote_code: bool = False) -> BackendSpec:
    if name == "fake":
        from ovbench.backends.fake import FakeBackend
        return BackendSpec("fake", device,
                           lambda cfg, i: FakeBackend(cfg, i),
                           lambda: (lambda t: len(t.split())))
    if name == "openvino":
        if not model:
            raise SystemExit("--model é obrigatório com o backend openvino.")
        from ovbench.backends.openvino_backend import OpenVINOBackend, tokenizer_counter
        return BackendSpec(
            "openvino", device,
            lambda cfg, i: OpenVINOBackend(model, cfg, device, enable_thinking, trust_remote_code, i),
            lambda: tokenizer_counter(model, trust_remote_code))
    raise SystemExit(f"Backend desconhecido: {name!r} (use 'openvino').")
