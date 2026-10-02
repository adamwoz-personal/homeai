# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Switch the GPU between Jarvis (voice) and the coding model.

    homeai-mode coding     # Jarvis offline; qwen3-coder gets the whole card
    homeai-mode voice      # back to normal: Jarvis on, coder shares the card
    homeai-mode status
    homeai-mode memory [off|on|retention DAYS|purge]   # see memory_privacy.py

Why: the voice model (7.0 GB) and qwen3-coder-30b (~19.7 GB) together need
~26 GB on a 21.4 GB card. Sharing, the coder generates at ~25 tok/s; with the
card to itself, q4_0 KV and every expert on the GPU, 131 tok/s
(tools/bench_kv_configs.sh, ROUTING_GUIDE Session 14).

Coding mode is runtime-only: its llama-server override lives in
/run/user/<uid>/systemd/user, which is cleared at reboot, and homeai.service
is stopped but not disabled. A reboot therefore always comes back in voice
mode, with a configuration that matches.

Standard library only. All side effects go through ``System`` so the
sequencing and rollback are unit-tested without touching the machine.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VOICE_SERVICE = "homeai.service"
CODER_SERVICE = "llama-server.service"
VOICE_MODEL = "llama31-voice"
OLLAMA = "http://127.0.0.1:11434"
CODER_HEALTH = "http://127.0.0.1:8080/health"
DROPIN_NAME = "homeai-coding-mode.conf"

# Measured fastest with the card to itself: 131 tok/s generation, 1543 tok/s
# prompt. q4_0 KV costs a little cache fidelity; --kv q8_0 --cpu-moe 6 gives
# 80 tok/s with the higher-fidelity cache.
DEFAULT_KV = "q4_0"
DEFAULT_CPU_MOE = 0

CODER_START_TIMEOUT_S = 300.0
UNLOAD_TIMEOUT_S = 30.0
LOAD_ATTEMPTS = 3


class ModeError(Exception):
    pass


def runtime_dropin_dir(uid: int | None = None) -> Path:
    uid = os.getuid() if uid is None else uid
    return Path(f"/run/user/{uid}/systemd/user/{CODER_SERVICE}.d")


def render_dropin(kv: str, cpu_moe: int, ctx: int | None) -> str:
    lines = [
        "# Written by homeai-mode coding; removed by homeai-mode voice and at reboot.",
        "[Service]",
        f"Environment=KV={kv}",
        f"Environment=NCPUMOE={cpu_moe}",
    ]
    if ctx:
        lines.append(f"Environment=CTX={ctx}")
    return "\n".join(lines) + "\n"


class System:
    """The real machine. Tests substitute a fake with the same methods."""

    def __init__(self, dropin_dir: Path | None = None) -> None:
        self.dropin_dir = dropin_dir or runtime_dropin_dir()

    def systemctl(self, *args: str) -> bool:
        proc = subprocess.run(["systemctl", "--user", *args], capture_output=True,
                              text=True, check=False, timeout=120)
        return proc.returncode == 0

    def is_active(self, unit: str) -> bool:
        return self.systemctl("is-active", "--quiet", unit)

    def http(self, url: str, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, str]:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except (urllib.error.URLError, OSError, TimeoutError):
            return 0, ""

    def dropin_exists(self) -> bool:
        return (self.dropin_dir / DROPIN_NAME).exists()

    def write_dropin(self, text: str) -> None:
        self.dropin_dir.mkdir(parents=True, exist_ok=True)
        (self.dropin_dir / DROPIN_NAME).write_text(text)

    def remove_dropin(self) -> None:
        (self.dropin_dir / DROPIN_NAME).unlink(missing_ok=True)

    def coder_cmdline(self) -> list[str]:
        proc = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value",
                               CODER_SERVICE], capture_output=True, text=True, check=False)
        pid = proc.stdout.strip()
        if not pid or pid == "0":
            return []
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except OSError:
            return []

    def vram_gb(self) -> tuple[float, float]:
        best = (0.0, 0.0)
        for total_file in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
            try:
                total = int(total_file.read_text()) / 1024**3
                used = int((total_file.parent / "mem_info_vram_used").read_text()) / 1024**3
            except (OSError, ValueError):
                continue
            if total > best[0]:
                best = (total, used)
        return best

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.monotonic()


@dataclass
class Status:
    mode: str
    jarvis: bool
    coder_up: bool
    coder_kv: str
    coder_cpu_moe: str
    loaded: list[str]
    vram_used_gb: float
    vram_total_gb: float

    def lines(self) -> list[str]:
        return [
            f"mode:          {self.mode}",
            f"jarvis:        {'running' if self.jarvis else 'stopped'} ({VOICE_SERVICE})",
            f"coder:         {'up' if self.coder_up else 'DOWN'}, KV {self.coder_kv or '?'}, "
            f"experts on CPU: {self.coder_cpu_moe or '0'}",
            f"ollama loaded: {', '.join(self.loaded) or 'nothing'}",
            f"vram:          {self.vram_used_gb:.1f} / {self.vram_total_gb:.1f} GB",
        ]


def _flag(cmdline: list[str], name: str) -> str:
    return cmdline[cmdline.index(name) + 1] if name in cmdline[:-1] else ""


class ModeSwitcher:
    def __init__(self, system: System, log=print) -> None:
        self.sys = system
        self.log = log

    # -- probes --------------------------------------------------------------

    def loaded_models(self) -> list[str]:
        code, body = self.sys.http(f"{OLLAMA}/api/ps")
        if code != 200:
            return []
        try:
            return [m.get("name", "") for m in json.loads(body).get("models", [])]
        except (ValueError, AttributeError):
            return []

    def voice_loaded(self) -> bool:
        return any(name.split(":")[0] == VOICE_MODEL for name in self.loaded_models())

    def coder_healthy(self) -> bool:
        code, _ = self.sys.http(CODER_HEALTH, timeout=3)
        return code == 200

    def status(self) -> Status:
        cmd = self.sys.coder_cmdline()
        total, used = self.sys.vram_gb()
        return Status(
            mode="coding" if self.sys.dropin_exists() else "voice",
            jarvis=self.sys.is_active(VOICE_SERVICE),
            coder_up=self.coder_healthy(),
            coder_kv=_flag(cmd, "--cache-type-k"),
            coder_cpu_moe=_flag(cmd, "--n-cpu-moe"),
            loaded=self.loaded_models(),
            vram_used_gb=used,
            vram_total_gb=total,
        )

    # -- steps ---------------------------------------------------------------

    def _wait(self, predicate, timeout_s: float, interval_s: float = 2.0) -> bool:
        deadline = self.sys.now() + timeout_s
        while True:
            if predicate():
                return True
            if self.sys.now() >= deadline:
                return False
            self.sys.sleep(interval_s)

    def _restart_coder(self) -> None:
        if not self.sys.systemctl("daemon-reload"):
            raise ModeError("systemctl --user daemon-reload failed")
        self.log(f"restarting {CODER_SERVICE} (loading ~17 GB of weights)")
        if not self.sys.systemctl("restart", CODER_SERVICE):
            raise ModeError(f"could not restart {CODER_SERVICE}")
        if not self._wait(self.coder_healthy, CODER_START_TIMEOUT_S):
            raise ModeError(f"{CODER_SERVICE} did not become healthy within "
                            f"{CODER_START_TIMEOUT_S:.0f}s; see journalctl --user -u {CODER_SERVICE}")

    def _unload_voice(self) -> None:
        if not self.voice_loaded():
            return
        self.log(f"unloading {VOICE_MODEL}")
        self.sys.http(f"{OLLAMA}/api/generate", {"model": VOICE_MODEL, "keep_alive": 0}, timeout=30)
        if not self._wait(lambda: not self.voice_loaded(), UNLOAD_TIMEOUT_S, 1.0):
            raise ModeError(f"{VOICE_MODEL} is still loaded in Ollama")

    def _load_voice(self) -> None:
        # Only after llama-server has finished allocating: a load issued while
        # it was still starting was measured to return without loading.
        for attempt in range(1, LOAD_ATTEMPTS + 1):
            if self.voice_loaded():
                return
            self.log(f"loading {VOICE_MODEL} (pinned), attempt {attempt}")
            self.sys.http(f"{OLLAMA}/api/generate", {"model": VOICE_MODEL, "keep_alive": -1},
                          timeout=120)
            if self.voice_loaded():
                return
            self.sys.sleep(3)
        raise ModeError(f"{VOICE_MODEL} did not load; Jarvis will try on the next question. "
                        "Check: ollama ps")

    # -- modes ---------------------------------------------------------------

    def coding(self, kv: str = DEFAULT_KV, cpu_moe: int = DEFAULT_CPU_MOE,
               ctx: int | None = None) -> None:
        self.log(f"stopping {VOICE_SERVICE} (Jarvis is offline in coding mode)")
        self.sys.systemctl("stop", VOICE_SERVICE)
        try:
            self._unload_voice()
            self.sys.write_dropin(render_dropin(kv, cpu_moe, ctx))
            self._restart_coder()
        except ModeError as exc:
            self.log(f"coding mode failed: {exc}")
            self.log("rolling back to voice mode")
            try:
                self.voice()
            except ModeError as rollback_exc:
                raise ModeError(f"{exc}; rollback also failed: {rollback_exc}") from exc
            raise ModeError(f"{exc} (rolled back to voice mode)") from exc
        if self.voice_loaded():
            self.log(f"WARNING: something reloaded {VOICE_MODEL}; the coder may be slow")
        self.log(f"coding mode: KV {kv}, experts on CPU {cpu_moe}"
                 + (f", context {ctx}" if ctx else ""))

    def voice(self) -> None:
        if self.sys.dropin_exists():
            self.sys.remove_dropin()
            self._restart_coder()
        elif not self.coder_healthy():
            self.log(f"{CODER_SERVICE} is down; starting it")
            self.sys.systemctl("start", CODER_SERVICE)
            if not self._wait(self.coder_healthy, CODER_START_TIMEOUT_S):
                self.log(f"WARNING: {CODER_SERVICE} is not healthy; continuing with Jarvis")
        self._load_voice()
        self.log(f"starting {VOICE_SERVICE}")
        if not self.sys.systemctl("start", VOICE_SERVICE):
            raise ModeError(f"could not start {VOICE_SERVICE}")
        self.sys.sleep(3)
        if not self.sys.is_active(VOICE_SERVICE):
            raise ModeError(f"{VOICE_SERVICE} did not stay up; journalctl --user -u homeai -n 50")
        self.log("voice mode: Jarvis is listening")


def _locked(path: Path):
    handle = open(path, "w")  # noqa: SIM115 - held for the process lifetime
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ModeError("another homeai-mode is already running") from None
    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="homeai-mode", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coding", help="give the coder the whole GPU (Jarvis offline)")
    c.add_argument("--kv", default=DEFAULT_KV, choices=["q4_0", "q8_0", "f16"],
                   help=f"KV cache type (default {DEFAULT_KV}, fastest measured)")
    c.add_argument("--cpu-moe", type=int, default=DEFAULT_CPU_MOE,
                   help="MoE layers kept on the CPU (q8_0 needs 6 to fit at 64K)")
    c.add_argument("--ctx", type=int, help="context tokens (default: llm-serve.sh's 65536)")
    sub.add_parser("voice", help="Jarvis back on; the coder shares the GPU")
    sub.add_parser("status")
    m = sub.add_parser("memory", help="what ZeroClaw keeps of what is said to it")
    m.add_argument("--config-dir", type=Path, help="ZeroClaw config dir (default ~/.zeroclaw)")
    msub = m.add_subparsers(dest="memory_cmd")
    msub.add_parser("status", help="saving on/off, retention, rows stored, disk used")
    msub.add_parser("on", help="save conversations (ZeroClaw default)")
    msub.add_parser("off", help="stop saving conversations (existing ones stay until purged)")
    r = msub.add_parser("retention", help="days to keep saved conversations")
    r.add_argument("days", type=int, help="0 = keep forever (ZeroClaw default 30)")
    pg = msub.add_parser("purge", help="delete all saved conversations now")
    pg.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    if args.cmd == "memory":
        return _memory_main(args)

    switcher = ModeSwitcher(System())
    try:
        if args.cmd == "status":
            print("\n".join(switcher.status().lines()))
            return 0
        lock = _locked(Path(f"/run/user/{os.getuid()}/homeai-mode.lock"))
        try:
            if args.cmd == "coding":
                switcher.coding(args.kv, args.cpu_moe, args.ctx)
            else:
                switcher.voice()
        finally:
            lock.close()
        print("\n".join(switcher.status().lines()))
        return 0
    except ModeError as exc:
        print(f"homeai-mode: {exc}", file=sys.stderr)
        return 1


def _memory_main(args, backend=None, ask=input) -> int:
    from homeai import memory_privacy as mp

    control = mp.MemoryControl(backend or mp.MemoryBackend(args.config_dir))
    cmd = args.memory_cmd or "status"
    try:
        if cmd in ("on", "off"):
            control.set_saving(cmd == "on")
        elif cmd == "retention":
            control.set_retention(args.days)
        elif cmd == "purge":
            def confirm(n: int) -> bool:
                try:
                    return ask(f"Delete {n} saved conversations permanently? [y/N] ") \
                        .strip().lower() in ("y", "yes")
                except EOFError:
                    return False
            control.purge(None if args.yes else confirm)
        print("\n".join(control.status().lines()))
        return 0
    except mp.MemoryControlError as exc:
        print(f"homeai-mode memory: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
