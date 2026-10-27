import json

import pytest

from ovbench.backends import make_spec
from ovbench.cli import main
from ovbench.hardware import CpuTopology, _filtrar_isa, _ler_faixa, detect_isa, detect_topology
from ovbench.metrics import Sample, aggregate, percentile, summarize
from ovbench.presets import baseline_preset, build_presets, parse_ov_config
from ovbench.runner import RunConfig, run_instance, run_interleaved, run_pool
from ovbench.validators import get_validator
from ovbench.workloads import build_length_ladder, load_workload

HIBRIDA = CpuTopology(model="Intel i9-14900K", vendor="GenuineIntel", logical=32, physical=24,
                      hybrid=True, p_logical=16, p_physical=8, e_cores=16)
COMUM = CpuTopology(model="Intel Xeon", vendor="GenuineIntel", logical=16, physical=8,
                    p_logical=16, p_physical=8)


def test_presets_hibrida_usa_contagens_reais():
    p = build_presets(HIBRIDA)
    assert p["pcore_latency"]["INFERENCE_NUM_THREADS"] == "16"
    assert p["pcore_sem_ht"]["INFERENCE_NUM_THREADS"] == "8"
    assert p["ecore_throughput"]["SCHEDULING_CORE_TYPE"] == "ECORE_ONLY"
    assert baseline_preset(HIBRIDA) == "pcore_latency"


def test_presets_nao_hibrida_sem_core_type():
    p = build_presets(COMUM)
    assert not any("SCHEDULING_CORE_TYPE" in c for c in p.values())
    assert baseline_preset(COMUM) == "latency_fisicos"


def test_presets_gpu():
    assert set(build_presets(COMUM, "GPU")) == {"default", "throughput"}


def test_parse_ov_config():
    assert parse_ov_config('{"A": 1}') == {"A": "1"}


def test_ler_faixa():
    assert _ler_faixa("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


def test_detect_topology_roda():
    t = detect_topology()
    assert t.logical >= 1 and t.physical >= 1


def test_percentil_e_resumo():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert summarize([1.0, 3.0])["mean"] == 2.0
    assert summarize([]) == {"n": 0}


def test_sample_decode_tps():
    s = Sample("a", 0, 10, 11, total_s=2.0, ttft_s=1.0)
    assert s.decode_tps == pytest.approx(10.0)
    assert Sample("a", 0, 10, 1, 1.0, 0.5).decode_tps is None
    assert aggregate([s])["decode_tok_s"]["mean"] == pytest.approx(10.0)


def test_validadores():
    assert get_validator("json")('```json\n{"a":1}\n```')
    assert not get_validator("json")("texto")
    assert get_validator("regex:^ok")("ok!")
    assert get_validator("none") is None


def test_workload_arquivo(tmp_path):
    f = tmp_path / "w.jsonl"
    f.write_text('{"id":"x","prompt":"oi","system":"s"}\n{"messages":[{"role":"user","content":"y"}]}\n')
    ps = load_workload(str(f))
    assert [p.id for p in ps] == ["x", "p2"] and ps[0].messages[0]["role"] == "system"


def test_escada_de_tamanhos():
    contar = lambda t: len(t.split())
    ps = build_length_ladder(contar, [50, 200])
    assert contar(ps[0].messages[0]["content"].split("\n\n", 1)[1]) >= 50
    assert ps[1].meta["alvo_tokens_entrada"] == 200


def test_run_instance_e_erro_isolado():
    spec = make_spec("fake", None)
    prompts = load_workload("geral")
    cfg = RunConfig(repetitions=2, warmup=1, max_new_tokens=8)
    ok = run_instance(spec.factory, "a", {}, prompts, cfg, get_validator("json"), progress=lambda *_: None)
    assert ok["status"] == "ok" and ok["resumo"]["latencia_ms"]["n"] == 8
    assert ok["resumo"]["formato_valido"] == 1.0
    ruim = run_instance(spec.factory, "b", {"FAIL": "1"}, prompts, cfg, progress=lambda *_: None)
    assert ruim["status"] == "erro"


def test_run_pool_mede_espera():
    spec = make_spec("fake", None)
    prompts = load_workload("geral")
    r = run_pool(spec.factory, "p", [{}], prompts, RunConfig(3, 0, 8), concorrencia=4,
                 progress=lambda *_: None)
    assert r["status"] == "ok" and r["requisicoes"] == 12
    assert r["resumo"]["espera_ms"]["max"] > 0  # 4 threads disputando 1 instância


def test_fixed_length():
    spec = make_spec("fake", None)
    r = run_instance(spec.factory, "a", {}, load_workload("curto"), RunConfig(1, 0, 8, fixed_length=37),
                     progress=lambda *_: None)
    assert r["resumo"]["tokens_saida"]["mean"] == 37


def test_cli_fim_a_fim(tmp_path, capsys):
    rc = main(["run", "--backend", "fake", "--repetitions", "1", "--warmup", "0",
               "--validator", "json", "--input-lengths", "30", "--save-responses", "--out", str(tmp_path)])
    assert rc == 0
    nomes = {p.suffix for p in tmp_path.iterdir()}
    assert {".json", ".md", ".csv", ".jsonl"} <= nomes
    rel = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert rel["ambiente"]["cpu"]["logical"] >= 1 and rel["resultados"]


def test_cli_pool(tmp_path):
    rc = main(["run", "--backend", "fake", "--mode", "pool", "--pool-instancias", "2",
               "--repetitions", "2", "--warmup", "0", "--out", str(tmp_path)])
    assert rc == 0


def test_filtrar_isa_normaliza_nomes():
    assert _filtrar_isa(["AVX2", "avx512-vnni", "amx_tile", "sse4_2"]) == ["avx2", "avx512_vnni", "amx_tile"]


def test_detect_isa_retorna_lista():
    assert isinstance(detect_isa(), list)


def test_filtrar_isa_formato_windows():
    flags = ["avx", "avx2", "avx512f", "avx512vnni", "amxtile", "avxvnni"]
    assert _filtrar_isa(flags) == ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_tile"]


def test_intercalado_distribui_rodadas_e_isola_erro():
    spec = make_spec("fake", None)
    presets = {"a": {}, "b": {}, "c": {}, "ruim": {"FAIL": "1"}}
    rs = run_interleaved(spec.factory, presets, load_workload("curto"), RunConfig(4, 0, 8),
                         progress=lambda *_: None)
    por_nome = {r["config"]: r for r in rs}
    assert por_nome["ruim"]["status"] == "erro"
    for n in "abc":
        assert por_nome[n]["status"] == "ok" and por_nome[n]["resumo"]["latencia_ms"]["n"] == 4
    ordens = por_nome["a"]["ordem_por_rodada"]
    assert len(ordens) == 4 and all(sorted(o) == ["a", "b", "c"] for o in ordens)


def test_intercalado_semente_reprodutivel():
    spec = make_spec("fake", None)
    f = lambda: run_interleaved(spec.factory, {"a": {}, "b": {}, "c": {}}, load_workload("curto"),
                                RunConfig(5, 0, 4), seed=7, progress=lambda *_: None)[0]["ordem_por_rodada"]
    assert f() == f()


def test_cli_interleave_shuffle_cooldown(tmp_path):
    base = ["run", "--backend", "fake", "--repetitions", "2", "--warmup", "0",
            "--presets", "default", "latency_fisicos", "--out", str(tmp_path)]
    assert main(base + ["--interleave", "--cooldown", "0.01"]) == 0
    assert main(base + ["--shuffle", "--cooldown", "0.01"]) == 0
