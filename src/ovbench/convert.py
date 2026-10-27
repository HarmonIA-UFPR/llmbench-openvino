"""Converte/quantiza um modelo do Hugging Face para OpenVINO IR (usa o optimum-cli)."""
from __future__ import annotations

import shutil
import subprocess
import sys


def converter(model_id: str, saida: str, weight_format: str = "int8", trust_remote_code: bool = False) -> int:
    exe = shutil.which("optimum-cli")
    cmd = ([exe] if exe else [sys.executable, "-m", "optimum.commands.optimum_cli"]) + [
        "export", "openvino", "--model", model_id, "--weight-format", weight_format]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(saida)
    print("Executando:", " ".join(cmd))
    return subprocess.call(cmd)
