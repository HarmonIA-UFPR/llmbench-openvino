"""Contrato mínimo de um backend de LLM para o benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    n_input_tokens: int
    n_output_tokens: int
    ttft_s: float | None  # tempo até o 1º token (inclui prefill); None se o backend não mede
    total_s: float        # tempo de geração (exclui tokenização/renderização do template)
    hit_limit: bool = False  # True se parou por max_new_tokens (resposta possivelmente cortada)


class LLMBackend(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def generate(
        self, messages: list[dict], max_new_tokens: int = 256, min_new_tokens: int | None = None,
    ) -> GenerationResult: ...

    def close(self) -> None:  # pragma: no cover - opcional
        pass
