#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Switch Jarvis's conversational model in one step, and back the same way (LLM_UPGRADE_PLAN).

Why: the voice model is set in one place, ZeroClaw's config (homeai/voice_model.py
reads it for pinning, the memory helper and the art patcher). Changing it by
hand means editing TOML, restarting ZeroClaw and getting the new model wholly
onto the GPU beside the coder, with Jarvis idle. This does all of that, and puts
everything back if ZeroClaw doesn't come up.

Checks first, changes nothing if any fail:
  * the model exists in Ollama with num_ctx 16384 (deploy/ollama/<name>.Modelfile);
  * nobody has spoken to Jarvis in the last 2 minutes (--force skips this);
  * the edited config parses, names the new model, and differs from the old one
    only in the voice provider's ``model`` and ``provider_extra``.
Then, with Jarvis paused (gpu_mode.ModeSwitcher.jarvis_paused): back up
config.toml, write it, restart ZeroClaw and wait for its health check; on
failure restore the backup and restart again. Jarvis comes back with the new
model reloaded wholly on the GPU.

``provider_extra = { reasoning_effort = "none" }`` is always set: it is the only
field that turns thinking off on Ollama's OpenAI route, and models without
thinking accept it (measured 2026-11-10).

    .venv/bin/python tools/switch_voice_model.py qwen35-voice [--dry-run]
    .venv/bin/python tools/switch_voice_model.py llama31-voice      # the way back
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai import service_scope, voice_model  # noqa: E402
from homeai.fsutil import atomic_write_text  # noqa: E402
from homeai.httpjson import OLLAMA, post_json  # noqa: E402

CTX = 16384
ZEROCLAW = "zeroclaw.service"
HEALTH = "http://127.0.0.1:42617/health"
IDLE_S = 120
EXTRA_LINE = 'provider_extra = { reasoning_effort = "none" }'


class SwitchError(Exception):
    pass


def check_model(model: str, post=post_json) -> None:
    try:
        params = post(f"{OLLAMA}/api/show", {"model": model}).get("parameters", "")
    except urllib.error.HTTPError as exc:
        raise SwitchError(f"Ollama has no {model}: ollama create {model} -f deploy/ollama/{model}.Modelfile") from exc
    if not re.search(rf"^num_ctx\s+{CTX}\s*$", params, re.M):
        raise SwitchError(f"{model} is not at num_ctx {CTX} (Jarvis's prompt would be cut); fix its Modelfile")


def edit(text: str, model: str) -> str:
    """The config with the voice provider's model replaced and thinking turned off; nothing else touched."""
    cfg = tomllib.loads(text)
    header = f"[providers.models.{voice_model.provider_ref(cfg)}]"
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration as exc:
        raise SwitchError(f"no {header} line to edit") from exc
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    body = lines[start + 1:end]
    model_at = [i for i, ln in enumerate(body) if re.match(r"\s*model\s*=", ln)]
    extra_at = [i for i, ln in enumerate(body) if re.match(r"\s*provider_extra\s*=", ln)]
    if len(model_at) != 1 or len(extra_at) > 1:
        raise SwitchError(f"{header} must have one model line and at most one provider_extra line; edit by hand")
    old_extra = voice_model.provider_section(cfg).get("provider_extra")
    if old_extra not in (None, voice_model.NO_REASONING):
        raise SwitchError(f"{header} has provider_extra {old_extra}; merge reasoning_effort by hand")
    body[model_at[0]] = f'model = "{model}"\n'
    if extra_at:
        body[extra_at[0]] = EXTRA_LINE + "\n"
    else:
        body.insert(model_at[0] + 1, EXTRA_LINE + "\n")
    new = "".join(lines[:start + 1] + body + lines[end:])
    verify(text, new, model)
    return new


def verify(old: str, new: str, model: str) -> None:
    a, b = tomllib.loads(old), tomllib.loads(new)
    sa, sb = voice_model.provider_section(a), voice_model.provider_section(b)
    if sb.get("model") != model or sb.get("provider_extra") != voice_model.NO_REASONING:
        raise SwitchError("the edit did not take")
    sa.pop("model", None), sa.pop("provider_extra", None), sb.pop("model"), sb.pop("provider_extra")
    if a != b:
        raise SwitchError("the edit changed more than the voice model; nothing written")


def recently_heard(run=subprocess.run) -> bool:
    argv = ["journalctl", "-u", "homeai", "--since", f"-{IDLE_S}s", "--no-pager", "-q"]
    if service_scope.scope() != "system":
        argv.insert(1, "--user")
    out = run(argv, capture_output=True, text=True, check=False).stdout
    return "heard:" in out


def healthy(post=post_json, sleep=time.sleep, tries: int = 30) -> bool:
    for _ in range(tries):
        try:
            post(HEALTH)
            return True
        except (OSError, ValueError):
            sleep(1)
    return False


def apply(path: Path, new: str, *, run=subprocess.run, ok=healthy, stamp: str | None = None, log=print) -> Path:
    """Back up, write, restart ZeroClaw; if it isn't healthy, put the old config back and restart again."""
    backup = path.with_name(f"{path.name}.bak-{stamp or time.strftime('%Y%m%d-%H%M%S')}")
    old = path.read_text(encoding="utf-8")
    backup.write_text(old, encoding="utf-8")
    atomic_write_text(path, new)
    run(service_scope.systemctl("restart", ZEROCLAW), check=False)
    if ok():
        log(f"zeroclaw up with the new model (backup: {backup})")
        return backup
    log("zeroclaw did not come up; restoring the old config")
    atomic_write_text(path, old)
    run(service_scope.systemctl("restart", ZEROCLAW), check=False)
    raise SwitchError(f"ZeroClaw failed with the new config; old config restored (journalctl --user -u zeroclaw)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--dry-run", action="store_true", help="show the config change, change nothing")
    ap.add_argument("--force", action="store_true", help="switch even if Jarvis was spoken to in the last 2 min")
    args = ap.parse_args()
    path = voice_model.config_path()
    try:
        before = voice_model.current(path)
        check_model(args.model)
        text = path.read_text(encoding="utf-8")
        new = edit(text, args.model)
        if new == text:
            print(f"Jarvis already uses {args.model} with thinking off")
            return 0
        print("".join(difflib.unified_diff(text.splitlines(True), new.splitlines(True), str(path), "new")))
        if args.dry_run:
            return 0
        if not args.force and recently_heard():
            raise SwitchError("someone spoke to Jarvis in the last 2 minutes; try later or pass --force")
        from homeai import gpu_mode as G
        with G.ModeSwitcher(G.System(), log=lambda line: print("[mode]", line, flush=True)).jarvis_paused():
            apply(path, new)
    except (SwitchError, ValueError) as exc:
        print(f"not switched: {exc}", file=sys.stderr)
        return 1
    print(f"Jarvis: {before} -> {voice_model.current(path)}. Way back: tools/switch_voice_model.py {before}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
