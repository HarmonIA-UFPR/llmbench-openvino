"""Cargas de teste: prompts embutidos genéricos, arquivos do usuário e escada de tamanhos."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Prompt:
    id: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)


BUILTIN: dict[str, list[tuple[str, str]]] = {
    "geral": [
        ("explicacao", "Explique em um parágrafo o que é aprendizado de máquina, para um iniciante."),
        ("resumo", "Resuma em 3 tópicos as vantagens de usar quantização INT8 em modelos de linguagem."),
        ("codigo", "Escreva uma função em Python que verifica se uma string é um palíndromo."),
        ("lista", "Liste cinco cuidados ao medir o desempenho de um modelo de linguagem em CPU."),
    ],
    "curto": [
        ("ola", "Responda apenas com uma frase curta: o que é uma CPU?"),
    ],
}


def load_workload(spec: str) -> list[Prompt]:
    """Nome embutido, .jsonl ({"prompt"|"messages", "system"?, "id"?}), .json (lista) ou .txt (1 prompt/linha)."""
    if spec in BUILTIN:
        return [Prompt(i, [{"role": "user", "content": t}]) for i, t in BUILTIN[spec]]
    caminho = Path(spec)
    if not caminho.is_file():
        raise SystemExit(f"Workload não encontrado: {spec!r} (embutidos: {', '.join(BUILTIN)})")
    texto = caminho.read_text(encoding="utf-8")
    if caminho.suffix == ".txt":
        itens = [{"prompt": l.strip()} for l in texto.splitlines() if l.strip()]
    elif caminho.suffix == ".json":
        itens = json.loads(texto)
    else:
        itens = [json.loads(l) for l in texto.splitlines() if l.strip()]
    prompts = []
    for n, item in enumerate(itens, 1):
        if "messages" in item:
            msgs = item["messages"]
        elif "prompt" in item:
            msgs = ([{"role": "system", "content": item["system"]}] if item.get("system") else []) \
                + [{"role": "user", "content": item["prompt"]}]
        else:
            raise SystemExit(f"Item {n} de {spec}: precisa de 'prompt' ou 'messages'.")
        prompts.append(Prompt(str(item.get("id", f"p{n}")), msgs, item.get("meta", {})))
    if not prompts:
        raise SystemExit(f"Workload vazio: {spec}")
    return prompts


_FRASE = ("O desempenho de modelos de linguagem em processadores depende da largura de banda de memória, "
          "do número de núcleos e da precisão numérica dos pesos. ")


def build_length_ladder(count_tokens: Callable[[str], int], alvos: list[int]) -> list[Prompt]:
    """Um prompt por alvo de tamanho de ENTRADA (≈ tokens do texto; o template soma poucos tokens)."""
    prompts = []
    palavras_base = _FRASE.split()
    for alvo in alvos:
        def texto(n: int) -> str:
            return " ".join((palavras_base * (n // len(palavras_base) + 1))[:n])

        lo, hi = 1, max(8, alvo * 4)
        while lo < hi:  # menor n com tokens >= alvo
            mid = (lo + hi) // 2
            if count_tokens(texto(mid)) >= alvo:
                hi = mid
            else:
                lo = mid + 1
        corpo = texto(lo)
        prompts.append(Prompt(f"entrada_{alvo}",
                              [{"role": "user", "content": "Resuma o texto a seguir.\n\n" + corpo}],
                              {"alvo_tokens_entrada": alvo}))
    return prompts
