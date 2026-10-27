"""Gera as configurações OpenVINO a comparar, a partir da topologia REAL da máquina.

Nada de contagem de threads fixa nem de PCORE_ONLY em CPU que não é híbrida.
"""
from __future__ import annotations

import json

from ovbench.hardware import CpuTopology


def build_presets(t: CpuTopology, device: str = "CPU") -> dict[str, dict]:
    if device.upper() != "CPU":
        # iGPU/NPU: sem noção de threads/pinagem de CPU.
        return {"default": {"PERFORMANCE_HINT": "LATENCY"},
                "throughput": {"PERFORMANCE_HINT": "THROUGHPUT"}}

    def cfg(threads: int, hint: str = "LATENCY", core_type: str | None = None,
            pin: bool = True, **extra) -> dict:
        c = {"INFERENCE_NUM_THREADS": str(max(1, threads)), "PERFORMANCE_HINT": hint}
        if pin:
            c["ENABLE_CPU_PINNING"] = "YES"
        if core_type:
            c["SCHEDULING_CORE_TYPE"] = core_type
        c.update({k: str(v) for k, v in extra.items()})
        return c

    p: dict[str, dict] = {"default": {"PERFORMANCE_HINT": "LATENCY"}}  # o que o OpenVINO escolhe sozinho
    if t.hybrid:
        p["pcore_latency"] = cfg(t.p_logical, core_type="PCORE_ONLY")
        p["pcore_sem_ht"] = cfg(t.p_physical, core_type="PCORE_ONLY")
        p["pcore_kv_cache_u8"] = cfg(t.p_logical, core_type="PCORE_ONLY", KV_CACHE_PRECISION="u8")
        p["pcore_throughput"] = cfg(t.p_logical, "THROUGHPUT", "PCORE_ONLY")
        if t.e_cores:
            p["ecore_throughput"] = cfg(t.e_cores, "THROUGHPUT", "ECORE_ONLY")
        p["todos_os_nucleos"] = cfg(t.logical, core_type="ANY_CORE")
    else:
        p["latency_fisicos"] = cfg(t.physical)
        p["latency_logicos"] = cfg(t.logical)
        p["kv_cache_u8"] = cfg(t.physical, KV_CACHE_PRECISION="u8")
        p["throughput"] = cfg(t.physical, "THROUGHPUT")
    p["sem_pinagem"] = cfg(t.physical, pin=False)
    return p


def baseline_preset(t: CpuTopology, device: str = "CPU") -> str:
    presets = build_presets(t, device)
    for nome in ("pcore_latency", "latency_fisicos", "default"):
        if nome in presets:
            return nome
    return next(iter(presets))


def parse_ov_config(spec: str) -> dict[str, str]:
    """'{"K":"v"}' ou caminho de um .json -> dict (valores viram str)."""
    texto = spec
    if not spec.lstrip().startswith("{"):
        with open(spec, encoding="utf-8") as f:
            texto = f.read()
    return {str(k): str(v) for k, v in json.loads(texto).items()}
