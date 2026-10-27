"""Relatórios: JSON (completo), CSV (uma linha por config) e Markdown (leitura humana)."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def _f(v, nd=1):
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def linha_resumo(r: dict) -> dict:
    base = {"config": r["config"], "modo": r["modo"], "status": r["status"]}
    if r["status"] != "ok":
        return {**base, "erro": r.get("erro", "")}
    s = r["resumo"]
    linha = {
        **base,
        "load_s": r.get("load_s"),
        "pico_rss_mb": r.get("pico_rss_mb"),
        "lat_media_ms": s["latencia_ms"].get("mean"),
        "lat_p50_ms": s["latencia_ms"].get("p50"),
        "lat_p95_ms": s["latencia_ms"].get("p95"),
        "ttft_medio_ms": s["ttft_ms"].get("mean"),
        "decode_tok_s": s["decode_tok_s"].get("mean"),
        "tok_s_geral": s["tok_s_geral"].get("mean"),
        "tokens_saida": s["tokens_saida"].get("mean"),
        "formato_valido": s["formato_valido"],
    }
    if r["modo"] == "pool":
        linha.update(e2e_p95_ms=s["e2e_ms"].get("p95"), espera_media_ms=s["espera_ms"].get("mean"),
                     throughput_req_s=r.get("throughput_req_s"), tokens_saida_s=r.get("tokens_saida_s"))
    return linha


def markdown(rel: dict) -> str:
    env, meta = rel["ambiente"], rel["meta"]
    cpu = env["cpu"]
    L = [f"# ovbench -- {meta['modelo']}", "",
         f"- Data: {meta['data']}  |  Backend: {meta['backend']}  |  Dispositivo: {meta['dispositivo']}",
         f"- CPU: {cpu['model']} ({cpu['logical']} lógicas / {cpu['physical']} físicas"
         + (f"; híbrida: {cpu['p_physical']}P/{cpu['p_logical']} thr + {cpu['e_cores']}E" if cpu["hybrid"] else "")
         + f") | RAM: {cpu.get('ram_gb')} GB | ISA: {', '.join(cpu['isa']) or '-'}",
         f"- SO: {env['os']} | Python {env['python']} | Pacotes: {env.get('pacotes', {})}",
         f"- Parâmetros: {meta['parametros']}", ""]
    linhas = [linha_resumo(r) for r in rel["resultados"]]
    ok = [l for l in linhas if l["status"] == "ok"]
    if ok:
        pool = ok[0]["modo"] == "pool"
        cab = ["Config", "Lat. média (ms)", "p95 (ms)", "TTFT (ms)", "Decode tok/s", "Tok. saída", "Pico RAM (MB)", "Formato"]
        if pool:
            cab += ["E2E p95 (ms)", "Espera (ms)", "req/s", "tok/s total"]
        L += ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
        for l in ok:
            c = [l["config"], _f(l["lat_media_ms"]), _f(l["lat_p95_ms"]), _f(l["ttft_medio_ms"]),
                 _f(l["decode_tok_s"], 2), _f(l["tokens_saida"], 0), _f(l["pico_rss_mb"], 0),
                 "-" if l["formato_valido"] is None else f"{l['formato_valido'] * 100:.0f}%"]
            if pool:
                c += [_f(l["e2e_p95_ms"]), _f(l["espera_media_ms"]), _f(l["throughput_req_s"], 2),
                      _f(l["tokens_saida_s"], 1)]
            L.append("| " + " | ".join(c) + " |")
    for l in linhas:
        if l["status"] != "ok":
            L.append(f"\n**{l['config']}**: falhou -- {l.get('erro')}")
    return "\n".join(L) + "\n"


def gravar(rel: dict, pasta: Path, prefixo: str) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = pasta / f"{prefixo}_{carimbo}"
    arquivos = []
    p = base.with_suffix(".json")
    p.write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    arquivos.append(p)
    p = base.with_suffix(".md")
    p.write_text(markdown(rel), encoding="utf-8")
    arquivos.append(p)
    linhas = [linha_resumo(r) for r in rel["resultados"]]
    campos = sorted({k for l in linhas for k in l}, key=lambda k: (k not in ("config", "modo", "status"), k))
    p = base.with_suffix(".csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    arquivos.append(p)
    return arquivos
