"""Validadores opcionais de formato da resposta (o benchmark mede velocidade, não qualidade)."""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
from typing import Callable

Validator = Callable[[str], bool]


def json_valido(texto: str) -> bool:
    t = texto.strip().replace("```json", "").replace("```", "").strip()
    if not t.startswith(("{", "[")):
        return False
    try:
        json.loads(t)
        return True
    except ValueError:
        return False


def nao_vazio(texto: str) -> bool:
    return bool(texto.strip())


def get_validator(spec: str | None) -> Validator | None:
    """None/'none' | 'json' | 'nonempty' | 'regex:<padrão>' | 'modulo.pacote:func' | 'arquivo.py:func'."""
    if not spec or spec == "none":
        return None
    if spec == "json":
        return json_valido
    if spec == "nonempty":
        return nao_vazio
    if spec.startswith("regex:"):
        padrao = re.compile(spec[6:], re.S)
        return lambda t: padrao.search(t) is not None
    if ":" in spec:
        origem, func = spec.rsplit(":", 1)
        if origem.endswith(".py"):
            s = importlib.util.spec_from_file_location("ovbench_validador_usuario", origem)
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
        else:
            mod = importlib.import_module(origem)
        return getattr(mod, func)
    raise SystemExit(f"Validador desconhecido: {spec!r}")
