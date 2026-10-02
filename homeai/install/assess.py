# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Assess whether this machine can run the voice assistant, and how.

Run before anything is installed:

    python3 -m homeai.install.assess            # human-readable report
    python3 -m homeai.install.assess --json     # for install.sh

Exit status 0 means "go" (the JSON carries the chosen tier); 2 means "abort",
with every blocking reason listed rather than only the first one found.

Standard library only: this runs under the system Python before the
project's virtualenv exists.

The thresholds are not guesses where they could be measured. On the
reference machine (RX 7900 XT, 21.4 GB VRAM) `ollama ps` reported the voice
model, llama3.1:8b, at 7.0 GB with num_ctx 16384 and 5.9 GB with 8192
(ROUTING_GUIDE.md Session 14). The CPU-only tier is NOT measured: it is
offered with a warning rather than silently.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024**3

# -- Thresholds ---------------------------------------------------------------
# Measured: voice model resident size at each context length, plus ~0.5 GB of
# headroom for the driver and display.
VRAM_FOR_16K_GB = 7.5
VRAM_FOR_8K_GB = 6.5
# A machine sold as "8 GB" reports ~7.6 GB in /proc/meminfo, so the floors are
# set slightly under the marketing number. GPU tier: whisper (~0.5 GB), Piper,
# the daemon, ZeroClaw and the desktop. CPU tier adds the 4.9 GB model and its
# KV cache to system RAM.
MIN_RAM_GPU_GB = 7.0
MIN_RAM_CPU_GB = 14.0
MIN_CORES_GPU = 4
MIN_CORES_CPU = 8
# Whisper runs on the CPU in every tier. small.en is the measured choice; on
# fewer cores it is too slow for conversation, so base.en is used instead.
CORES_FOR_SMALL_WHISPER = 8
# llama3.1:8b (4.9 GB) + whisper small.en (0.5 GB) + whisper.cpp build (~1 GB)
# + venv with onnxruntime/scipy (~1.5 GB) + Piper voice, with margin.
MIN_DISK_GB = 15.0
MIN_PYTHON = (3, 12)
SUPPORTED_ARCH = ("x86_64", "aarch64")

VOICE_BASE_MODEL = "llama3.1:8b"

# command -> how to get it. A compiler is checked separately (any one will do).
REQUIRED_COMMANDS: dict[str, str] = {
    "git": "sudo apt install git",
    "curl": "sudo apt install curl",
    "cmake": "sudo apt install cmake",
    "make": "sudo apt install build-essential",
    "ollama": "curl -fsSL https://ollama.com/install.sh | sh",
    "zeroclaw": "install ZeroClaw (>= 0.8) so that `zeroclaw` is on PATH",
}
COMPILERS = ("c++", "g++", "clang++")
COMPILER_HINT = "sudo apt install build-essential"


@dataclass
class Gpu:
    vendor: str
    name: str
    vram_total_gb: float
    vram_used_gb: float = 0.0


@dataclass
class Facts:
    os: str = ""
    arch: str = ""
    python: tuple[int, int] = (0, 0)
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpus: list[Gpu] = field(default_factory=list)
    capture_devices: int = 0
    playback_devices: int = 0
    commands: dict[str, bool] = field(default_factory=dict)
    systemd_user: bool = False

    @property
    def best_gpu(self) -> Gpu | None:
        return max(self.gpus, key=lambda g: g.vram_total_gb, default=None)

    @classmethod
    def from_dict(cls, data: dict) -> "Facts":
        data = dict(data)
        data["gpus"] = [Gpu(**g) for g in data.get("gpus", [])]
        if "python" in data:
            data["python"] = tuple(data["python"])[:2]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Tier:
    name: str
    model: str
    num_ctx: int
    whisper_model: str
    uses_gpu: bool
    summary: str


@dataclass
class Assessment:
    ok: bool
    tier: Tier | None
    blockers: list[str]
    warnings: list[str]
    facts: Facts

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tier": asdict(self.tier) if self.tier else None,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "facts": asdict(self.facts),
        }


# -- Parsers (pure, unit-tested) ---------------------------------------------


def parse_meminfo(text: str) -> tuple[float, float]:
    """Return (total, available) in GiB from /proc/meminfo content."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0])  # kB
    total = values.get("MemTotal", 0) * 1024 / GIB
    avail = values.get("MemAvailable", values.get("MemFree", 0)) * 1024 / GIB
    return round(total, 1), round(avail, 1)


def parse_asound_pcm(text: str) -> tuple[int, int]:
    """Count (capture, playback) PCM devices from /proc/asound/pcm.

    Lines look like ``00-00: ALC897 Analog : ALC897 Analog : playback 1 : capture 1``.
    """
    capture = playback = 0
    for line in text.splitlines():
        lowered = line.lower()
        if "capture" in lowered:
            capture += 1
        if "playback" in lowered:
            playback += 1
    return capture, playback


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,memory.used
    --format=csv,noheader,nounits`` (values in MiB)."""
    gpus = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total, used = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        gpus.append(Gpu("nvidia", parts[0], round(total / 1024, 1), round(used / 1024, 1)))
    return gpus


def read_amd_sysfs(drm_root: Path = Path("/sys/class/drm")) -> list[Gpu]:
    """AMD exposes VRAM in sysfs; no ROCm tooling is needed to read it."""
    gpus = []
    for total_file in sorted(drm_root.glob("card*/device/mem_info_vram_total")):
        device = total_file.parent
        try:
            total = int(total_file.read_text().strip())
            used_file = device / "mem_info_vram_used"
            used = int(used_file.read_text().strip()) if used_file.exists() else 0
        except (OSError, ValueError):
            continue
        name = device.parent.name
        gpus.append(Gpu("amd", name, round(total / GIB, 1), round(used / GIB, 1)))
    return gpus


# -- Probes (touch the real machine) -----------------------------------------


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def probe(disk_path: Path) -> Facts:
    total, avail = parse_meminfo(_read("/proc/meminfo"))
    capture, playback = parse_asound_pcm(_read("/proc/asound/pcm"))
    gpus = read_amd_sysfs()
    if shutil.which("nvidia-smi"):
        gpus += parse_nvidia_smi(
            _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                  "--format=csv,noheader,nounits"])
        )
    probe_path = disk_path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        disk_free = shutil.disk_usage(probe_path).free / GIB
    except OSError:
        disk_free = 0.0
    commands = {cmd: shutil.which(cmd) is not None for cmd in REQUIRED_COMMANDS}
    commands["compiler"] = any(shutil.which(c) for c in COMPILERS)
    return Facts(
        os=platform.system(),
        arch=platform.machine(),
        python=sys.version_info[:2],
        cpu_cores=os.cpu_count() or 0,
        ram_total_gb=total,
        ram_available_gb=avail,
        disk_free_gb=round(disk_free, 1),
        gpus=gpus,
        capture_devices=capture,
        playback_devices=playback,
        commands=commands,
        # `systemctl --user is-system-running` exits non-zero when merely
        # "degraded", so test for the user manager's runtime directory instead.
        systemd_user=bool(shutil.which("systemctl"))
        and Path(f"/run/user/{os.getuid()}/systemd").exists(),
    )


# -- Decision ----------------------------------------------------------------


def choose_tier(facts: Facts) -> Tier | None:
    whisper = "small.en" if facts.cpu_cores >= CORES_FOR_SMALL_WHISPER else "base.en"
    gpu = facts.best_gpu
    vram = gpu.vram_total_gb if gpu else 0.0
    if vram >= VRAM_FOR_16K_GB and facts.ram_total_gb >= MIN_RAM_GPU_GB:
        return Tier("gpu-16k", VOICE_BASE_MODEL, 16384, whisper, True,
                    f"{VOICE_BASE_MODEL} on the GPU with a 16K context (the reference setup)")
    if vram >= VRAM_FOR_8K_GB and facts.ram_total_gb >= MIN_RAM_GPU_GB:
        return Tier("gpu-8k", VOICE_BASE_MODEL, 8192, whisper, True,
                    f"{VOICE_BASE_MODEL} on the GPU with an 8K context "
                    "(enough for a voice request, ~4K tokens)")
    if facts.ram_total_gb >= MIN_RAM_CPU_GB and facts.cpu_cores >= MIN_CORES_CPU:
        return Tier("cpu-8k", VOICE_BASE_MODEL, 8192, whisper, False,
                    f"{VOICE_BASE_MODEL} on the CPU with an 8K context (slow; unmeasured)")
    return None


def assess(facts: Facts, *, require_audio: bool = True) -> Assessment:
    blockers: list[str] = []
    warnings: list[str] = []

    if facts.os != "Linux":
        blockers.append(f"Linux is required (found {facts.os or 'unknown'}): audio "
                        "discovery, systemd units and the GPU probes are Linux-specific.")
    if facts.arch not in SUPPORTED_ARCH:
        blockers.append(f"CPU architecture {facts.arch or 'unknown'} is not supported "
                        f"(need one of {', '.join(SUPPORTED_ARCH)}).")
    if tuple(facts.python) < MIN_PYTHON:
        blockers.append(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required "
                        f"(found {facts.python[0]}.{facts.python[1]}).")
    if facts.disk_free_gb < MIN_DISK_GB:
        blockers.append(f"Not enough free disk: {facts.disk_free_gb:.1f} GB free, "
                        f"{MIN_DISK_GB:.0f} GB needed (voice model 4.9 GB, speech "
                        "models, whisper.cpp build, Python environment).")

    for name, hint in REQUIRED_COMMANDS.items():
        if not facts.commands.get(name, False):
            blockers.append(f"Missing `{name}`. To fix: {hint}")
    if not facts.commands.get("compiler", False):
        blockers.append(f"Missing a C++ compiler (needed to build whisper.cpp). "
                        f"To fix: {COMPILER_HINT}")

    if facts.capture_devices == 0:
        msg = ("No microphone (ALSA capture device) found. A voice assistant cannot "
               "work without one; plug one in and re-run.")
        (blockers if require_audio else warnings).append(msg)
    if facts.playback_devices == 0:
        msg = "No speaker (ALSA playback device) found; replies could not be heard."
        (blockers if require_audio else warnings).append(msg)

    tier = choose_tier(facts)
    if tier is None:
        gpu = facts.best_gpu
        vram = f"{gpu.vram_total_gb:.1f} GB" if gpu else "no GPU"
        blockers.append(
            "Not enough resources to run the conversational model. Need EITHER a GPU "
            f"with >= {VRAM_FOR_8K_GB} GB VRAM and >= {MIN_RAM_GPU_GB:.0f} GB RAM, OR "
            f">= {MIN_RAM_CPU_GB:.0f} GB RAM and >= {MIN_CORES_CPU} CPU threads for "
            f"CPU-only. Found: {vram}, {facts.ram_total_gb:.1f} GB RAM, "
            f"{facts.cpu_cores} threads. Smaller models were tested and could not "
            "hold a conversation (they ignore the persona and hedge), so a weaker "
            "install is refused rather than offered."
        )
    else:
        if facts.cpu_cores < MIN_CORES_GPU:
            blockers.append(f"At least {MIN_CORES_GPU} CPU threads are needed for "
                            f"speech recognition (found {facts.cpu_cores}).")
        if tier.whisper_model != "small.en":
            warnings.append(f"Only {facts.cpu_cores} CPU threads: using the faster, less "
                            "accurate whisper base.en for speech recognition.")
        if not tier.uses_gpu:
            warnings.append("No GPU with enough VRAM: the model runs on the CPU. Expect "
                            "several seconds before each reply (this tier is unmeasured).")
        else:
            gpu = facts.best_gpu
            assert gpu is not None
            free = gpu.vram_total_gb - gpu.vram_used_gb
            need = VRAM_FOR_16K_GB if tier.num_ctx == 16384 else VRAM_FOR_8K_GB
            if free < need:
                warnings.append(
                    f"GPU {gpu.name} has {gpu.vram_total_gb:.1f} GB but {gpu.vram_used_gb:.1f} "
                    f"GB is in use right now. Another model (or an existing install) may "
                    "share it; if the voice model will not fit it spills to the CPU and "
                    "slows down. Check `ollama ps` after install."
                )
            if gpu.vendor == "amd":
                warnings.append("AMD GPU: Ollama needs ROCm or Vulkan support for this "
                                "card. The installer checks `ollama ps` reports GPU use.")
        if facts.ram_available_gb and facts.ram_available_gb < 3.0:
            warnings.append(f"Only {facts.ram_available_gb:.1f} GB RAM is free right now; "
                            "close something before first use.")

    if not facts.systemd_user:
        warnings.append("No systemd user session: the service will not be installed; "
                        "run `.venv/bin/python -m homeai.daemon` by hand.")

    return Assessment(not blockers, tier, blockers, warnings, facts)


def format_report(result: Assessment) -> str:
    f = result.facts
    gpu = f.best_gpu
    lines = [
        "Home AI machine assessment",
        f"  OS/arch     {f.os} {f.arch}, Python {f.python[0]}.{f.python[1]}",
        f"  CPU         {f.cpu_cores} threads",
        f"  RAM         {f.ram_total_gb:.1f} GB total, {f.ram_available_gb:.1f} GB free",
        f"  GPU         " + (f"{gpu.vendor} {gpu.name}: {gpu.vram_total_gb:.1f} GB VRAM "
                             f"({gpu.vram_used_gb:.1f} GB in use)" if gpu else "none usable"),
        f"  Disk        {f.disk_free_gb:.1f} GB free",
        f"  Audio       {f.capture_devices} capture, {f.playback_devices} playback devices",
    ]
    if result.tier:
        t = result.tier
        lines.append(f"  Choice      {t.name}: {t.summary}; whisper {t.whisper_model}")
    for w in result.warnings:
        lines.append(f"  WARNING     {w}")
    if result.ok:
        lines.append("Result: OK to install.")
    else:
        lines.append("Result: CANNOT INSTALL. Reasons:")
        lines.extend(f"  - {b}" for b in result.blockers)
    return "\n".join(lines)


def write_env_file(result: Assessment, path: Path) -> None:
    """Write the decision as shell assignments for install.sh to `source`.

    Values are shlex-quoted, and only the tier is written: nothing derived from
    device names or other free text reaches the shell.
    """
    import shlex

    tier = result.tier
    values = {
        "ASSESS_OK": "1" if result.ok else "0",
        "TIER_NAME": tier.name if tier else "",
        "VOICE_BASE_MODEL": tier.model if tier else "",
        "NUM_CTX": str(tier.num_ctx) if tier else "",
        "WHISPER_MODEL": tier.whisper_model if tier else "",
        "USES_GPU": "1" if tier and tier.uses_gpu else "0",
        "SYSTEMD_USER": "1" if result.facts.systemd_user else "0",
    }
    path.write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in values.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--env-file", type=Path,
                        help="also write the decision as shell assignments here")
    parser.add_argument("--disk-path", type=Path, default=Path.cwd(),
                        help="where models and the build will live (default: cwd)")
    parser.add_argument("--facts", type=Path,
                        help="JSON file of facts to use instead of probing (testing)")
    parser.add_argument("--no-audio-required", action="store_true",
                        help="treat a missing mic/speaker as a warning (headless prep)")
    args = parser.parse_args(argv)

    if args.facts:
        facts = Facts.from_dict(json.loads(args.facts.read_text()))
    else:
        facts = probe(args.disk_path)
    result = assess(facts, require_audio=not args.no_audio_required)
    if args.env_file:
        write_env_file(result, args.env_file)
    print(json.dumps(result.to_dict(), indent=2) if args.json else format_report(result))
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
