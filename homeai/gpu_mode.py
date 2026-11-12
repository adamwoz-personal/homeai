# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Switch the GPU between Jarvis (voice) and the coding model.

    homeai-mode coding     # Jarvis offline; qwen3-coder gets the whole card
    homeai-mode voice      # back to normal: Jarvis on, coder shares the card
    homeai-mode art        # ComfyUI replaces the coder; Jarvis stays on
    homeai-mode status
    homeai-mode memory [off|on|retention DAYS|purge]   # see memory_privacy.py
    homeai-mode speaker [on|off|fetch|remove|enrol|identify]   # see speaker_admin.py

Why: the voice model (7.0 GB) and qwen3-coder-30b (~19.7 GB) together need
~26 GB on a 21.4 GB card. Sharing, the coder generates at ~25 tok/s; with the
card to itself, q4_0 KV and every expert on the GPU, 131 tok/s
(tools/bench_kv_configs.sh, ROUTING_GUIDE Session 14).

Coding mode is runtime-only: its llama-server override lives in
/run/user/<uid>/systemd/user, which is cleared at reboot, and homeai.service
is stopped but not disabled. A reboot therefore always comes back in voice
mode, with a configuration that matches.

Art mode is runtime-only too: ComfyUI runs as a transient unit (systemd-run),
which a reboot forgets. It is capped at 24 GB of RAM with no swap, because an
uncapped image model once pushed the box into swap (tools/art, 2026-11-05).

Standard library only. All side effects go through ``System`` so the
sequencing and rollback are unit-tested without touching the machine.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from homeai import gpu_info, service_scope, voice_model
from homeai.httpjson import OLLAMA

VOICE_SERVICE = "homeai.service"
CODER_SERVICE = "llama-server.service"


def _voice() -> str:
    """Jarvis's model, from ZeroClaw's config (homeai/voice_model.py): never a copy that could disagree."""
    return voice_model.current()
CODER_HEALTH = "http://127.0.0.1:8080/health"
DROPIN_NAME = "homeai-coding-mode.conf"
COMFY_UNIT = "homeai-comfyui.service"
COMFY_HEALTH = "http://127.0.0.1:8188/system_stats"
CODER_SLOTS = "http://127.0.0.1:8080/slots"
COMFY_ROOT = Path.home() / ".local/share/homeai/comfyui"
COMFY_MEMORY_MAX = "24G"
COMFY_START_TIMEOUT_S = 180.0

# Measured fastest with the card to itself: 131 tok/s generation, 1543 tok/s
# prompt. q4_0 KV costs a little cache fidelity; --kv q8_0 --cpu-moe 6 gives
# 80 tok/s with the higher-fidelity cache.
DEFAULT_KV = "q4_0"
DEFAULT_CPU_MOE = 0

# Coding-mode (KV, experts on CPU) per coder preset (~/.config/llm-preset).
# qwen36 (22.4 GB) does not fit with every expert on the GPU; 6 measured best,
# 83 tok/s gen / 1148 prompt (2026-11-10, ~/homeai-bench/kv.jsonl). A preset
# not listed gets qwen36's: offloading too much is slow, too little fails to load.
PRESET_FILE = Path.home() / ".config/llm-preset"
CODING_DEFAULTS = {"qwen": (DEFAULT_KV, DEFAULT_CPU_MOE), "qwen36": ("q8_0", 6)}


def coding_defaults(preset_file: Path | None = None) -> tuple[str, int]:
    """Measured coding-mode KV type and CPU expert layers for the coder in use."""
    try:
        preset = (preset_file or PRESET_FILE).read_text(encoding="utf-8").strip() or "qwen"
    except FileNotFoundError:
        preset = "qwen"   # llm-serve's own fallback
    return CODING_DEFAULTS.get(preset, CODING_DEFAULTS["qwen36"])

CODER_START_TIMEOUT_S = 300.0
UNLOAD_TIMEOUT_S = 30.0
LOAD_ATTEMPTS = 3
PAUSE_RELOADS = 6        # after a benchmark: tries to get the voice model wholly back on the GPU


class ModeError(Exception):
    pass


def comfy_command(root: Path = COMFY_ROOT) -> list[str]:
    """ComfyUI as a transient, memory-capped user unit. No --cache-none: it reloads the model for every image."""
    return ["systemd-run", "--user", "--quiet", "--collect", f"--unit={COMFY_UNIT}",
            "-p", f"MemoryMax={COMFY_MEMORY_MAX}", "-p", "MemorySwapMax=0",
            "-p", f"WorkingDirectory={root / 'app'}", "-E", "HIP_VISIBLE_DEVICES=0",
            str(root / "venv/bin/python"), "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--disable-auto-launch", "--disable-pinned-memory"]


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
        proc = subprocess.run(service_scope.systemctl(*args), capture_output=True,
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

    def start_comfy(self) -> bool:
        proc = subprocess.run(comfy_command(), capture_output=True, text=True, check=False, timeout=60)
        return proc.returncode == 0

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
        card = gpu_info.biggest()
        return (card.total / 1024**3, card.used / 1024**3) if card else (0.0, 0.0)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.monotonic()


@dataclass
class Status:
    mode: str
    jarvis: bool
    coder_up: bool
    comfy_up: bool
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
            f"comfyui:       {'running' if self.comfy_up else 'stopped'} ({COMFY_UNIT})",
            f"ollama loaded: {', '.join(self.loaded) or 'nothing'}",
            f"vram:          {self.vram_used_gb:.1f} / {self.vram_total_gb:.1f} GB",
        ]


def _flag(cmdline: list[str], name: str) -> str:
    return cmdline[cmdline.index(name) + 1] if name in cmdline[:-1] else ""


def _exclusive(method):
    """One mode switch at a time, across processes (``lock_path``) and threads. Re-entrant, so a failed
    switch can roll back through ``voice()`` while it still holds the lock."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._mutex:
            if self._depth == 0 and self.lock_path is not None:
                self._handle = _locked(self.lock_path, self.lock_wait_s)
            self._depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._depth -= 1
                if self._depth == 0 and self._handle is not None:
                    self._handle.close()
                    self._handle = None
    return wrapper


class ModeSwitcher:
    def __init__(self, system: System, log=print, lock_path: Path | None = None, lock_wait_s: float = 0.0) -> None:
        """``lock_path``: the file every switcher locks (mode_lock_path() for real use; None in unit tests).
        ``lock_wait_s``: how long to wait for another switch to finish (voice: VOICE_LOCK_WAIT_S; CLI: 0)."""
        self.sys = system
        self.log = log
        self.lock_path = lock_path
        self.lock_wait_s = lock_wait_s
        self._mutex = threading.RLock()
        self._depth = 0
        self._handle = None

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
        return any(name.split(":")[0] == _voice() for name in self.loaded_models())

    def coder_healthy(self) -> bool:
        code, _ = self.sys.http(CODER_HEALTH, timeout=3)
        return code == 200

    def coder_busy(self) -> bool:
        """True while the coder is answering someone; a missing or odd answer counts as idle."""
        code, body = self.sys.http(CODER_SLOTS, timeout=5)
        if code != 200:
            return False
        try:
            return any(slot.get("is_processing") for slot in json.loads(body))
        except (ValueError, AttributeError, TypeError):
            return False

    def comfy_running(self) -> bool:
        return self.sys.is_active(COMFY_UNIT)

    def mode(self) -> str:
        if self.sys.dropin_exists():
            return "coding"
        return "art" if self.comfy_running() else "voice"

    def status(self) -> Status:
        cmd = self.sys.coder_cmdline()
        total, used = self.sys.vram_gb()
        return Status(
            mode=self.mode(),
            jarvis=self.sys.is_active(VOICE_SERVICE),
            coder_up=self.coder_healthy(),
            comfy_up=self.comfy_running(),
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
        self.log(f"unloading {_voice()}")
        self.sys.http(f"{OLLAMA}/api/generate", {"model": _voice(), "keep_alive": 0}, timeout=30)
        if not self._wait(lambda: not self.voice_loaded(), UNLOAD_TIMEOUT_S, 1.0):
            raise ModeError(f"{_voice()} is still loaded in Ollama")

    def _load_voice(self) -> None:
        # Only after llama-server has finished allocating: a load issued while
        # it was still starting was measured to return without loading.
        for attempt in range(1, LOAD_ATTEMPTS + 1):
            if self.voice_loaded():
                return
            self.log(f"loading {_voice()} (pinned), attempt {attempt}")
            self.sys.http(f"{OLLAMA}/api/generate", {"model": _voice(), "keep_alive": -1},
                          timeout=120)
            if self.voice_loaded():
                return
            self.sys.sleep(3)
        raise ModeError(f"{_voice()} did not load; Jarvis will try on the next question. "
                        "Check: ollama ps")

    def _stop_comfy(self) -> None:
        if not self.comfy_running():
            return
        self.log(f"stopping {COMFY_UNIT}")
        self.sys.systemctl("stop", COMFY_UNIT)
        if self.comfy_running():
            raise ModeError(f"{COMFY_UNIT} would not stop; systemctl --user status {COMFY_UNIT}")

    def _comfy_healthy(self) -> bool:
        code, _ = self.sys.http(COMFY_HEALTH, timeout=3)
        return code == 200

    # -- modes ---------------------------------------------------------------

    @_exclusive
    def art(self) -> None:
        """ComfyUI takes the coder's place on the card; Jarvis stays on."""
        if self.sys.dropin_exists():
            raise ModeError("in coding mode; run homeai-mode voice first")
        if self.comfy_running():
            self.log("already in art mode")
            return
        if self.coder_busy():
            raise ModeError("the coder is busy; try again when it's idle")
        self.log(f"stopping {CODER_SERVICE} (art mode needs its VRAM)")
        self.sys.systemctl("stop", CODER_SERVICE)
        try:
            self.log(f"starting {COMFY_UNIT} (capped at {COMFY_MEMORY_MAX} RAM, no swap)")
            if not self.sys.start_comfy():
                raise ModeError(f"could not start {COMFY_UNIT}")
            if not self._wait(self._comfy_healthy, COMFY_START_TIMEOUT_S):
                raise ModeError(f"ComfyUI did not answer within {COMFY_START_TIMEOUT_S:.0f}s; "
                                f"see journalctl --user -u {COMFY_UNIT}")
        except ModeError as exc:
            self.log(f"art mode failed: {exc}")
            self.log("rolling back to voice mode")
            try:
                self.voice()            # stops ComfyUI first
            except ModeError as rollback_exc:
                raise ModeError(f"{exc}; rollback also failed: {rollback_exc}") from exc
            raise ModeError(f"{exc} (rolled back to voice mode)") from exc
        if not self.voice_loaded():
            self.log(f"WARNING: {_voice()} is not loaded; Jarvis loads it on the next question")
        self.log("art mode: ComfyUI on 127.0.0.1:8188, Jarvis listening")

    @_exclusive
    def restart_comfy(self) -> None:
        """A fresh ComfyUI in art mode, e.g. after a job timed out (its job thread may be dead)."""
        if self.mode() != "art":
            raise ModeError("not in art mode")
        self._stop_comfy()
        if not self.sys.start_comfy():
            raise ModeError(f"could not start {COMFY_UNIT}")
        if not self._wait(self._comfy_healthy, COMFY_START_TIMEOUT_S):
            raise ModeError(f"ComfyUI did not answer within {COMFY_START_TIMEOUT_S:.0f}s after a restart")
        self.log("ComfyUI restarted")

    @_exclusive
    def coding(self, kv: str = DEFAULT_KV, cpu_moe: int = DEFAULT_CPU_MOE,
               ctx: int | None = None) -> None:
        self._stop_comfy()
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
            self.log(f"WARNING: something reloaded {_voice()}; the coder may be slow")
        self.log(f"coding mode: KV {kv}, experts on CPU {cpu_moe}"
                 + (f", context {ctx}" if ctx else ""))

    def voice_on_gpu(self) -> bool:
        """The voice model is loaded and none of it sits on the CPU (image jobs pushed 13% there, 2026-11-05)."""
        code, body = self.sys.http(f"{OLLAMA}/api/ps")
        try:
            models = json.loads(body).get("models", []) if code == 200 else []
        except (ValueError, AttributeError):
            return False
        return any(m.get("name", "").split(":")[0] == _voice() and m.get("size_vram") == m.get("size")
                   for m in models)

    @contextlib.contextmanager
    def coder_paused(self, voice_off: bool = False):
        """The coder stopped for a benchmark's ``with`` block (tools that run their own ComfyUI).

        Only from voice mode, and not while the coder is answering someone. ``voice_off`` also stops
        Jarvis and unloads the voice model, for image models that need the whole card (Adam agreed).
        Afterwards, whatever happened in the block: the coder, then the voice model reloaded until it
        sits wholly on the GPU, then Jarvis.
        """
        mode = self.mode()
        if mode != "voice":
            raise ModeError(f"in {mode} mode; benchmarks start from voice mode")
        if self.coder_busy():
            raise ModeError("the coder is busy; try again when it's idle")
        self.log(f"stopping {CODER_SERVICE} for the benchmark")
        self.sys.systemctl("stop", CODER_SERVICE)
        try:
            if voice_off:
                self.log(f"stopping {VOICE_SERVICE} (this model needs the whole card)")
                self.sys.systemctl("stop", VOICE_SERVICE)
                self._unload_voice()
            yield self
        finally:
            self._resume_after_pause()

    def _resume_after_pause(self) -> None:
        try:
            self.log(f"starting {CODER_SERVICE}")
            self.sys.systemctl("start", CODER_SERVICE)
            if not self._wait(self.coder_healthy, CODER_START_TIMEOUT_S):
                self.log(f"WARNING: {CODER_SERVICE} is not healthy; the voice model loads on Jarvis's next question")
                return
            self._reload_voice_on_gpu()
        finally:
            self.log(f"starting {VOICE_SERVICE}")
            self.sys.systemctl("start", VOICE_SERVICE)

    @contextlib.contextmanager
    def jarvis_paused(self):
        """Jarvis stopped for a voice-model eval's ``with`` block (tools/eval_tool_misfire.py).

        Candidates take the voice model's place on the card; a question to Jarvis meanwhile would
        reload it and skew the timings. The coder stays up: that is the card the candidates must share.
        Afterwards, whatever happened: the voice model reloaded until wholly on the GPU, then Jarvis.
        """
        mode = self.mode()
        if mode != "voice":
            raise ModeError(f"in {mode} mode; evals start from voice mode")
        self.log(f"stopping {VOICE_SERVICE} for the eval")
        self.sys.systemctl("stop", VOICE_SERVICE)
        try:
            yield self
        finally:
            try:
                self._reload_voice_on_gpu()
            finally:
                self.log(f"starting {VOICE_SERVICE}")
                self.sys.systemctl("start", VOICE_SERVICE)

    def _reload_voice_on_gpu(self) -> None:
        """Reload the voice model until none of it sits on the CPU (PAUSE_RELOADS tries)."""
        for attempt in range(1, PAUSE_RELOADS + 1):
            # Reload even if loaded: it may sit partly on the CPU. The coder can still be claiming VRAM
            # after /health answers, so a load can fail; wait and try again.
            try:
                self._unload_voice()
                self.sys.sleep(3)
                self._load_voice()
            except ModeError as exc:
                self.log(f"voice model reload {attempt}: {exc}")
            if self.voice_on_gpu():
                return
            self.sys.sleep(10)
        self.log(f"WARNING: {_voice()} is not wholly on the GPU; check: ollama ps")

    @_exclusive
    def voice(self) -> None:
        self._stop_comfy()
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
            raise ModeError(f"{VOICE_SERVICE} did not stay up; {service_scope.journal_hint()}")
        self.log("voice mode: Jarvis is listening")


def mode_lock_path(uid: int | None = None) -> Path:
    """Shared by the CLI and the daemon's art voice. In the runtime dir the sandboxed daemon may write
    (homeai.service ReadWritePaths), not /run/user/<uid> itself, which is read-only to it."""
    return Path(f"/run/user/{os.getuid() if uid is None else uid}/homeai/mode.lock")


VOICE_LOCK_WAIT_S = 120   # a switch takes up to about a minute (art mode: ComfyUI start)
LOCK_POLL_S = 0.25


def _locked(path: Path, wait_s: float = 0.0, *, sleep=time.sleep, clock=time.monotonic):
    """Take the mode lock. ``wait_s`` > 0: wait that long for a running switch to end (voice, Adam
    2026-11-12: a voice request waits and then wins); 0: refuse at once (the CLI)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = open(path, "a")  # noqa: SIM115 - held until the switch ends
    deadline = clock() + wait_s
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if clock() >= deadline:
                handle.close()
                raise ModeError("another mode switch is already running") from None
            sleep(LOCK_POLL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="homeai-mode", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coding", help="give the coder the whole GPU (Jarvis offline)")
    c.add_argument("--kv", choices=["q4_0", "q8_0", "f16"],
                   help="KV cache type (default: measured best for the coder preset)")
    c.add_argument("--cpu-moe", type=int,
                   help="MoE layers kept on the CPU (default: measured best for the coder preset)")
    c.add_argument("--ctx", type=int, help="context tokens (default: llm-serve.sh's 65536)")
    sub.add_parser("voice", help="Jarvis back on; the coder shares the GPU")
    sub.add_parser("art", help="ComfyUI replaces the coder for drawing; Jarvis stays on")
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
    from homeai import speaker_admin
    speaker_admin.add_parser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "memory":
        return _memory_main(args)
    if args.cmd == "speaker":
        return speaker_admin.main(args)

    switcher = ModeSwitcher(System(), lock_path=mode_lock_path())
    try:
        if args.cmd == "status":
            print("\n".join(switcher.status().lines()))
            return 0
        if args.cmd == "coding":
            kv, cpu_moe = coding_defaults()
            switcher.coding(args.kv or kv, cpu_moe if args.cpu_moe is None else args.cpu_moe, args.ctx)
        elif args.cmd == "art":
            switcher.art()
        else:
            switcher.voice()
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
