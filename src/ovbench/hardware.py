"""Detecção da topologia da CPU Intel (híbrida P/E-core ou não) e do ambiente de execução."""
from __future__ import annotations

import os
import platform
import re
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CpuTopology:
    model: str = "desconhecido"
    vendor: str = "desconhecido"
    logical: int = 1
    physical: int = 1
    hybrid: bool = False
    p_logical: int = 0    # threads lógicas nos P-cores (== logical se não híbrida)
    p_physical: int = 0   # P-cores físicos
    e_cores: int = 0      # E-cores (cada um com 1 thread)
    isa: list[str] = field(default_factory=list)
    ram_gb: float | None = None
    virtualized: bool = False     # WSL/VM: a topologia real costuma ficar escondida
    source: str = "detectado"      # "detectado" ou "informado" (--cpu-topology)
    avisos: list[str] = field(default_factory=list)

    @property
    def is_intel(self) -> bool:
        return "intel" in (self.vendor + " " + self.model).lower()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_intel"] = self.is_intel
        return d


def _ler_faixa(texto: str) -> list[int]:
    """'0-15,20' -> [0..15, 20]."""
    saida: list[int] = []
    for parte in texto.strip().split(","):
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-")
            saida.extend(range(int(a), int(b) + 1))
        else:
            saida.append(int(parte))
    return saida


def _hibrida_linux() -> tuple[int, int, int] | None:
    base = Path("/sys/devices")
    try:
        p_cpus = _ler_faixa((base / "cpu_core" / "cpus").read_text())
        e_cpus = _ler_faixa((base / "cpu_atom" / "cpus").read_text())
    except OSError:
        return None
    if not p_cpus or not e_cpus:
        return None
    nucleos = set()
    for c in p_cpus:
        try:
            nucleos.add((base / "system" / "cpu" / f"cpu{c}" / "topology" / "core_id").read_text().strip())
        except OSError:
            nucleos.add(str(c))
    return len(p_cpus), len(nucleos), len(e_cpus)


def _hibrida_windows() -> tuple[int, int, int] | None:
    """Usa GetSystemCpuSetInformation: EfficiencyClass maior = P-core."""
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetSystemCpuSetInformation.restype = wintypes.BOOL
        necessario = wintypes.ULONG(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(necessario), None, 0)
        if necessario.value == 0:
            return None
        buf = ctypes.create_string_buffer(necessario.value)
        if not k32.GetSystemCpuSetInformation(buf, necessario, ctypes.byref(necessario), None, 0):
            return None
        raw = buf.raw[: necessario.value]
        off = 0
        por_classe: dict[int, set] = {}
        logicos: dict[int, int] = {}
        while off + 20 <= len(raw):
            tamanho, tipo = struct.unpack_from("<II", raw, off)
            if tamanho == 0:
                break
            if tipo == 0:  # CpuSetInformation
                grupo = struct.unpack_from("<H", raw, off + 12)[0]
                nucleo = raw[off + 15]
                classe = raw[off + 18]
                por_classe.setdefault(classe, set()).add((grupo, nucleo))
                logicos[classe] = logicos.get(classe, 0) + 1
            off += tamanho
        if len(por_classe) < 2:
            return None
        p = max(por_classe)
        e_logicos = sum(v for k, v in logicos.items() if k != p)
        return logicos[p], len(por_classe[p]), e_logicos
    except Exception:
        return None


def _modelo_vendor() -> tuple[str, str]:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return (winreg.QueryValueEx(k, "ProcessorNameString")[0].strip(),
                        winreg.QueryValueEx(k, "VendorIdentifier")[0].strip())
        except OSError:
            pass
    try:
        modelo = vendor = None
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name") and modelo is None:
                modelo = linha.split(":", 1)[1].strip()
            elif linha.startswith("vendor_id") and vendor is None:
                vendor = linha.split(":", 1)[1].strip()
            if modelo and vendor:
                return modelo, vendor
    except OSError:
        pass
    return platform.processor() or "desconhecido", "desconhecido"


_ISA_ALVO = ["avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_tile", "amx_int8", "amx_bf16"]


def _filtrar_isa(flags) -> list[str]:
    """Mantém só as flags relevantes, ignorando '_' e '-' (Linux: avx512_vnni; Windows: avx512vnni)."""
    def chave(f) -> str:
        return str(f).lower().replace("_", "").replace("-", "")

    presentes = {chave(f) for f in flags}
    return [f for f in _ISA_ALVO if chave(f) in presentes]


def _isa_linux() -> list[str]:
    try:
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("flags"):
                return _filtrar_isa(linha.split(":", 1)[1].split())
    except OSError:
        pass
    return []


def _isa_cpuinfo() -> list[str]:
    """Via py-cpuinfo (CPUID de verdade; funciona no Windows e no macOS)."""
    try:
        import cpuinfo
        return _filtrar_isa(cpuinfo.get_cpu_info().get("flags", []))
    except Exception:
        return []


def _isa_windows_api() -> list[str]:
    """Fallback sem dependências: IsProcessorFeaturePresent (só AVX2 e AVX-512F)."""
    try:
        import ctypes
        presente = ctypes.windll.kernel32.IsProcessorFeaturePresent
        saida = []
        if presente(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE
            saida.append("avx2")
        if presente(41):  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
            saida.append("avx512f")
        return saida
    except Exception:
        return []


def detect_isa() -> list[str]:
    if sys.platform.startswith("linux"):
        isa = _isa_linux()
        if isa:
            return isa
    isa = _isa_cpuinfo()
    if not isa and sys.platform == "win32":
        isa = _isa_windows_api()
    return isa


_PADRAO_TOPOLOGIA = re.compile(r"^\s*(\d+)\s*P(?:\s*\+\s*(\d+)\s*E)?\s*(?:,\s*(ht|noht))?\s*$", re.I)


def parse_topology_override(spec: str) -> dict:
    """'8P+4E' (P-cores com HT), '8P+4E,noht', '6P' (sem E-cores). Devolve contagens."""
    m = _PADRAO_TOPOLOGIA.match(spec or "")
    if not m:
        raise SystemExit(f"--cpu-topology inválido: {spec!r}. Exemplos: 8P+4E | 8P+4E,noht | 6P")
    p, e = int(m.group(1)), int(m.group(2) or 0)
    ht = (m.group(3) or "ht").lower() == "ht"
    p_logical = p * (2 if ht else 1)
    return {"p_physical": p, "p_logical": p_logical, "e_cores": e,
            "logical": p_logical + e, "physical": p + e, "hybrid": e > 0}


def _virtualizado() -> bool:
    if sys.platform.startswith("linux"):
        try:
            if "microsoft" in Path("/proc/version").read_text().lower():
                return True  # WSL
            for linha in Path("/proc/cpuinfo").read_text().splitlines():
                if linha.startswith("flags"):
                    return "hypervisor" in linha.split()
        except OSError:
            pass
    return False


_MODELO_COSTUMA_SER_HIBRIDO = re.compile(r"(\b(1[2-9]|[2-9]\d)th Gen\b|Core\(TM\) Ultra)", re.I)


def detect_topology(override: str | None = None) -> CpuTopology:
    """`override` (ou a variável OVBENCH_CPU_TOPOLOGY) sobrepõe a contagem de núcleos detectada."""
    modelo, vendor = _modelo_vendor()
    logical = os.cpu_count() or 1
    physical = logical
    ram = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    hib = _hibrida_windows() if sys.platform == "win32" else _hibrida_linux()
    t = CpuTopology(model=modelo, vendor=vendor, logical=logical, physical=physical,
                    isa=detect_isa(), ram_gb=ram, virtualized=_virtualizado())
    if hib:
        t.hybrid, t.p_logical, t.p_physical, t.e_cores = True, hib[0], hib[1], hib[2]
    else:
        t.p_logical, t.p_physical = logical, physical

    override = override or os.environ.get("OVBENCH_CPU_TOPOLOGY")
    if override:
        for k, v in parse_topology_override(override).items():
            setattr(t, k, v)
        t.source = "informado"
        if t.logical != (os.cpu_count() or t.logical):
            t.avisos.append(
                f"A topologia informada ({override}) soma {t.logical} threads, mas o sistema vê "
                f"{os.cpu_count()}. Confira se o valor está certo.")
    elif not t.hybrid and _MODELO_COSTUMA_SER_HIBRIDO.search(modelo):
        onde = " (WSL/máquina virtual esconde a topologia real)" if t.virtualized else ""
        t.avisos.append(
            f"O modelo '{modelo}' costuma ser uma CPU híbrida (P+E-cores), mas a topologia não foi "
            f"detectada{onde}; por isso os presets de P-core/E-core não foram gerados e as contagens de "
            "núcleos podem estar erradas. Informe a topologia com --cpu-topology (ex.: 8P+4E) ou rode no "
            "Windows/Linux nativo. Alguns modelos dessa geração (ex.: i3) não são híbridos.")
    return t


def detect_environment(override: str | None = None) -> dict:
    """Ficha completa do ambiente, gravada em todo relatório."""
    env: dict = {"cpu": detect_topology(override).to_dict(), "os": platform.platform(),
                 "python": platform.python_version()}
    for pacote in ("openvino", "optimum", "optimum.intel", "transformers", "torch"):
        try:
            from importlib.metadata import version
            env.setdefault("pacotes", {})[pacote] = version(pacote.replace(".", "-"))
        except Exception:
            pass
    try:
        import openvino as ov
        core = ov.Core()
        env["dispositivos_openvino"] = {
            d: str(core.get_property(d, "FULL_DEVICE_NAME")) for d in core.available_devices}
    except Exception:
        env["dispositivos_openvino"] = {}
    return env
