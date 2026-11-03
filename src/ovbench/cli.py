"""Linha de comando: ovbench info | run | convert."""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from ovbench import __version__
from ovbench.backends import make_spec
from ovbench.hardware import detect_environment, detect_topology
from ovbench.presets import baseline_preset, build_presets, parse_ov_config
from ovbench.report import gravar, markdown
from ovbench.runner import RunConfig, run_instance, run_interleaved, run_pool
from ovbench.validators import get_validator
from ovbench.workloads import build_length_ladder, load_workload


def _bool(v):
    return None if v is None else v == "true"


def cmd_info(args) -> int:
    t = detect_topology(args.cpu_topology)
    print(json.dumps(detect_environment(args.cpu_topology), ensure_ascii=False, indent=2))
    for aviso in t.avisos:
        print(f"\n[aviso] {aviso}")
    if not t.is_intel:
        print("\n[aviso] CPU não identificada como Intel; o OpenVINO funciona, mas os presets foram pensados para Intel.")
    print(f"\nPresets disponíveis para {args.device}:")
    for nome, cfg in build_presets(t, args.device).items():
        print(f"  {nome}: {cfg}")
    return 0


def cmd_download(args) -> int:
    from ovbench.download import baixar
    destino = baixar(args.repo_id, args.out, args.revision)
    print(f"Modelo em: {destino}")
    print(f"Próximo passo: ovbench run --model {destino}")
    return 0


def cmd_convert(args) -> int:
    from ovbench.convert import converter
    return converter(args.model_id, args.out, args.weight_format, args.trust_remote_code)


def _exigir_modelo(args) -> None:
    """Para cedo, com instruções, se a pasta do modelo não existe (evita repetir o erro em cada preset)."""
    if args.backend != "openvino":
        return
    pasta = Path(args.model).expanduser()
    if not pasta.is_dir():
        raise SystemExit(
            f"Pasta do modelo não encontrada: {pasta.resolve()}\n"
            "Baixe um modelo já em OpenVINO IR:  ovbench download --repo-id OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov\n"
            "ou converta um do Hugging Face:      ovbench convert --model-id <id> --out <pasta>\n"
            "(a pasta modelos/ não vai para o git: cada máquina precisa baixar o seu).")
    if not any(pasta.glob("*.xml")):
        raise SystemExit(f"Nenhum arquivo .xml em {pasta.resolve()}: a pasta não parece conter um modelo OpenVINO IR.")


def cmd_run(args) -> int:
    topo = detect_topology(args.cpu_topology)
    for aviso in topo.avisos:
        print(f"[aviso] {aviso}")
    spec = make_spec(args.backend, args.model, args.device, _bool(args.enable_thinking), args.trust_remote_code)
    todos = build_presets(topo, args.device)
    for extra in args.ov_config or []:
        nome, _, valor = extra.partition("=")
        if not valor:
            raise SystemExit("--ov-config espera NOME=JSON_OU_ARQUIVO")
        todos[nome] = parse_ov_config(valor)

    if args.list_presets:
        for n, c in todos.items():
            print(f"{n}: {c}")
        return 0

    _exigir_modelo(args)
    prompts = load_workload(args.workload)
    if args.input_lengths:
        prompts += build_length_ladder(spec.counter(), args.input_lengths)
    validator = get_validator(args.validator)
    cfg = RunConfig(args.repetitions, args.warmup, args.max_new_tokens, args.fixed_length)

    nomes = args.presets or ([baseline_preset(topo, args.device)] if args.mode == "pool" else list(todos))
    faltando = [n for n in nomes if n not in todos]
    if faltando:
        raise SystemExit(f"Preset(s) desconhecido(s): {faltando}. Use --list-presets.")

    modelo = args.model or args.backend
    pasta = Path(args.out)
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saida = open(pasta / f"respostas_{carimbo}.jsonl", "w", encoding="utf-8") if args.save_responses else None
    print(f"ovbench {__version__} | modelo: {modelo} | CPU: {topo.model} | "
          f"{'híbrida' if topo.hybrid else 'não híbrida'} | prompts: {len(prompts)} x {cfg.repetitions} rep.")
    resultados = []
    try:
        if args.mode == "instancia" and args.interleave:
            print(f"\n--- intercalado: {', '.join(nomes)} (todos carregados ao mesmo tempo; semente {args.seed})")
            resultados += run_interleaved(spec.factory, {n: todos[n] for n in nomes}, prompts, cfg,
                                          validator, saida, args.cooldown, args.seed)
        elif args.mode == "instancia":
            if args.shuffle:
                nomes = random.Random(args.seed).sample(nomes, len(nomes))
            for i, nome in enumerate(nomes):
                if args.cooldown and i:
                    print(f"  (pausa de {args.cooldown:g}s para esfriar)")
                    time.sleep(args.cooldown)
                print(f"\n--- {nome}: {todos[nome]}")
                r = run_instance(spec.factory, nome, todos[nome], prompts, cfg, validator, saida)
                if r["status"] == "ok":
                    print(f"  lat. média {r['resumo']['latencia_ms']['mean']:.0f} ms | "
                          f"decode {r['resumo']['decode_tok_s'].get('mean', float('nan')):.1f} tok/s")
                resultados.append(r)
        else:
            nome = nomes[0]
            cfgs = [todos[n] for n in nomes] if len(nomes) > 1 else [todos[nome]] * args.pool_instancias
            rotulo = "+".join(nomes) if len(nomes) > 1 else f"{nome} x{args.pool_instancias}"
            print(f"\n--- pool '{rotulo}' ({len(cfgs)} instância(s)), concorrência {args.pool_concorrencia or len(cfgs)}")
            resultados.append(run_pool(spec.factory, rotulo, cfgs, prompts, cfg,
                                       args.pool_concorrencia, validator, saida))
    finally:
        if saida:
            saida.close()

    rel = {
        "meta": {"modelo": modelo, "backend": spec.name, "dispositivo": args.device, "versao": __version__,
                 "data": datetime.now().isoformat(timespec="seconds"),
                 "parametros": {k: v for k, v in vars(args).items() if k != "func"}},
        "ambiente": detect_environment(args.cpu_topology),
        "resultados": resultados,
    }
    print("\n" + markdown(rel))
    prefixo = "ovbench_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(modelo).name)
    for a in gravar(rel, pasta, prefixo):
        print("Gravado:", a)
    return 0 if any(r["status"] == "ok" for r in resultados) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ovbench", description="Benchmark de LLMs em OpenVINO (Intel)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="mostra hardware detectado e presets gerados")
    p.add_argument("--device", default="CPU")
    p.add_argument("--cpu-topology", metavar="NP+NE",
                   help="informa a topologia à mão, ex.: 8P+4E (P-cores com HT), 8P+4E,noht, 6P; "
                        "também aceita a variável OVBENCH_CPU_TOPOLOGY")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("download", help="baixa do Hugging Face um modelo já em formato OpenVINO IR")
    p.add_argument("--repo-id", required=True, help="ex.: OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov")
    p.add_argument("--out", help="pasta de destino (padrão: modelos/<nome do repositório>)")
    p.add_argument("--revision")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("convert", help="converte/quantiza um modelo do Hugging Face para OpenVINO")
    p.add_argument("--model-id", required=True, help="id do Hugging Face ou pasta local")
    p.add_argument("--out", required=True)
    p.add_argument("--weight-format", default="int8", choices=["fp16", "int8", "int4"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("run", help="roda o benchmark")
    p.add_argument("--model", help="pasta do modelo em formato OpenVINO IR")
    p.add_argument("--backend", default="openvino", choices=["openvino", "fake"])
    p.add_argument("--device", default="CPU", help="CPU (padrão), GPU ou NPU")
    p.add_argument("--cpu-topology", metavar="NP+NE",
                   help="informa a topologia à mão, ex.: 8P+4E (P-cores com HT), 8P+4E,noht, 6P; "
                        "também aceita a variável OVBENCH_CPU_TOPOLOGY")
    p.add_argument("--mode", choices=["instancia", "pool"], default="instancia")
    p.add_argument("--presets", nargs="+", help="presets a testar (padrão: todos; no pool, 1 = baseline)")
    p.add_argument("--ov-config", action="append", metavar="NOME=JSON",
                   help="preset extra: JSON inline ou arquivo .json (repetível)")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--workload", default="geral", help="geral | curto | arquivo .jsonl/.json/.txt")
    p.add_argument("--input-lengths", type=int, nargs="+", help="acrescenta prompts de ~N tokens de entrada")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--fixed-length", type=int, help="gera exatamente N tokens (comparação justa entre modelos)")
    p.add_argument("--validator", default="none", help="none | json | nonempty | regex:PAD | mod:func | arq.py:func")
    p.add_argument("--enable-thinking", choices=["true", "false"], help="para modelos híbridos (ex.: Qwen3)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--interleave", action="store_true",
                   help="intercala as repetições entre os presets (neutraliza deriva térmica); "
                        "mantém todos carregados ao mesmo tempo")
    p.add_argument("--shuffle", action="store_true", help="embaralha a ordem dos presets (modo sequencial)")
    p.add_argument("--cooldown", type=float, default=0.0, help="pausa em segundos entre blocos, para esfriar a CPU")
    p.add_argument("--seed", type=int, default=42, help="semente da ordem embaralhada")
    p.add_argument("--pool-instancias", type=int, default=2)
    p.add_argument("--pool-concorrencia", type=int)
    p.add_argument("--save-responses", action="store_true")
    p.add_argument("--out", default="resultados")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
