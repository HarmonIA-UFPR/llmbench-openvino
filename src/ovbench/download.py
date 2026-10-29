"""Baixa um modelo já em formato OpenVINO IR do Hugging Face (ex.: OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov)."""
from __future__ import annotations

from pathlib import Path


def baixar(repo_id: str, saida: str | None = None, revision: str | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # vem junto com transformers (extra [openvino])
        raise SystemExit("huggingface_hub não instalado: pip install -e '.[openvino]'") from e
    destino = Path(saida or Path("modelos") / repo_id.split("/")[-1]).expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    caminho = snapshot_download(repo_id=repo_id, local_dir=str(destino), revision=revision)
    if not any(Path(caminho).glob("*.xml")):
        print(f"[aviso] nenhum arquivo .xml em {caminho}: o repositório pode não estar em formato OpenVINO IR.")
    return Path(caminho)
