#!/usr/bin/env bash
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
#
# Home AI installer.
#
#   ./install.sh --dry-run     # assess and show every action, change nothing
#   ./install.sh               # install
#
# What it does, in order. Every step is skipped if already done, so re-running
# is safe, and nothing the user owns is overwritten without a backup.
#
#   1. Assess the machine (homeai/install/assess.py) and choose the model tier.
#      ABORTS with every reason listed if the machine cannot run a home AI.
#   2. Python environment (.venv) with the voice extras.
#   3. whisper.cpp built from source, plus the speech-recognition model.
#   4. The Piper voice, and the speaker-recognition model (off until
#      `homeai-mode speaker on`; see homeai/speaker_admin.py).
#   5. Ollama: pull llama3.1:8b and create the `llama31-voice` variant with the
#      tier's context length. The VRAM guard needs sudo: see --sudo.
#   6. ZeroClaw: add the locked-down voice agent and copy its persona.
#   7. .env and the systemd user service.
#   8. Verify: `python -m homeai.daemon --check` and one real agent reply.
#
# Options:
#   --dry-run              print actions instead of performing them
#   --sudo                 also install the Ollama VRAM guard (asks for a password)
#   --no-service           do not install or start the systemd user service
#   --no-audio-required    allow installing on a machine without mic/speaker
#   --agent NAME           ZeroClaw agent name (default: HOMEAI_AGENT_NAME in .env, else jarvis)
#   --zeroclaw-dir DIR     ZeroClaw config dir (default: ~/.zeroclaw)
#   --skip-verify          skip step 8 (it loads the model, ~10s)
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$REPO"

DRY_RUN=0
USE_SUDO=0
INSTALL_SERVICE=1
AUDIO_FLAG=""
AGENT=""
ZC_DIR="${HOME}/.zeroclaw"
VERIFY=1

usage() { sed -n '4,31p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --sudo) USE_SUDO=1 ;;
        --no-service) INSTALL_SERVICE=0 ;;
        --no-audio-required) AUDIO_FLAG="--no-audio-required" ;;
        --agent) AGENT="${2:?--agent needs a name}"; shift ;;
        --zeroclaw-dir) ZC_DIR="${2:?--zeroclaw-dir needs a path}"; shift ;;
        --skip-verify) VERIFY=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 64 ;;
    esac
    shift
done

# -- helpers ----------------------------------------------------------------
# Never `cmd | grep -q` here: with pipefail, grep -q exits on the first match,
# the writer dies of SIGPIPE and the whole test reads as false.

STEP=0
NOTES=()
step() { STEP=$((STEP + 1)); printf '\n== %d. %s\n' "$STEP" "$*"; }
info() { printf '   %s\n' "$*"; }
note() { NOTES+=("$*"); printf '   NOTE: %s\n' "$*"; }
die()  { printf '\nINSTALL ABORTED: %s\n' "$*" >&2; exit 1; }

# Run a command, or only show it in a dry run.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '   would run: %s\n' "$*"
    else
        printf '   $ %s\n' "$*"
        "$@"
    fi
}

# mkdir only what is missing, so a re-run executes nothing.
ensure_dir() {
    local d
    for d in "$@"; do [[ -d "$d" ]] || run mkdir -p "$d"; done
}

# Append KEY=VALUE to .env unless KEY is already set there (the user's value wins).
env_set_default() {
    local key="$1" value="$2"
    if [[ -f .env ]] && grep -q "^${key}=" .env; then
        info ".env already sets ${key}; leaving it"
        return
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        info "would add to .env: ${key}=${value}"
    else
        [[ -f .env ]] || { install -m 600 /dev/null .env; }
        printf '%s=%s\n' "$key" "$value" >> .env
        info "added to .env: ${key}=${value}"
    fi
}

# -- 1. assess ----------------------------------------------------------------

step "Assessing this machine"
command -v python3 >/dev/null || die "python3 is not installed (need 3.12+): sudo apt install python3 python3-venv"

ASSESS_ENV="$(mktemp)"
trap 'rm -f "$ASSESS_ENV"' EXIT
set +e
# HOMEAI_ASSESS_FACTS: a JSON facts file used instead of probing (tests only).
FACTS_ARGS=()
[[ -n "${HOMEAI_ASSESS_FACTS:-}" ]] && FACTS_ARGS=(--facts "$HOMEAI_ASSESS_FACTS")
python3 -m homeai.install.assess --disk-path "$REPO" --env-file "$ASSESS_ENV" $AUDIO_FLAG "${FACTS_ARGS[@]}"
ASSESS_RC=$?
set -e
if [[ $ASSESS_RC -ne 0 ]]; then
    die "this machine cannot run Home AI as it is (reasons above). Nothing was changed."
fi
# shellcheck disable=SC1090  # written by assess.py with shlex-quoted values
source "$ASSESS_ENV"
info "tier: ${TIER_NAME} (${VOICE_BASE_MODEL}, context ${NUM_CTX}, whisper ${WHISPER_MODEL})"

if [[ -z "$AGENT" ]]; then
    if [[ -f .env ]] && grep -q '^HOMEAI_AGENT_NAME=' .env; then
        AGENT="$(grep '^HOMEAI_AGENT_NAME=' .env | tail -1 | cut -d= -f2-)"
    else
        AGENT="jarvis"
    fi
fi
[[ "$AGENT" =~ ^[A-Za-z0-9_-]+$ ]] || die "agent name '$AGENT' must be letters, digits, - or _"
info "ZeroClaw agent: ${AGENT} (config dir ${ZC_DIR})"

# -- 2. python environment ---------------------------------------------------

step "Python environment"
if [[ -x .venv/bin/python ]]; then
    info ".venv exists"
elif command -v uv >/dev/null; then
    # The exact interpreter that passed the assessment: uv's own lookup of
    # "python3" can find an older one first (seen: 3.11 in ~/.local/bin).
    run uv venv --python "$(command -v python3)" .venv
else
    run python3 -m venv .venv
fi
if command -v uv >/dev/null; then
    run uv pip install --python .venv/bin/python -q -e '.[voice,dev]'
else
    run .venv/bin/python -m pip install -q -e '.[voice,dev]'
fi

# -- 3. speech recognition ------------------------------------------------------

step "Speech recognition (whisper.cpp, model ${WHISPER_MODEL})"
WHISPER_DIR="vendor/whisper.cpp"
WHISPER_REF="v1.9.4"   # tested; the reference install is v1.9.4 plus later commits
if [[ -d "$WHISPER_DIR/.git" ]]; then
    info "$WHISPER_DIR present"
else
    run git clone --depth 1 --branch "$WHISPER_REF" https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
fi
# OpenBLAS speeds whisper up on the CPU; build without it rather than fail.
if ldconfig -p 2>/dev/null | grep libopenblas >/dev/null; then
    BUILD_DIR="build-blas"; CMAKE_EXTRA=(-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS)
else
    BUILD_DIR="build"; CMAKE_EXTRA=()
    note "OpenBLAS not found; whisper is built without it (slower). For speed: sudo apt install libopenblas-dev, delete $WHISPER_DIR/build and re-run."
fi
WHISPER_BIN="$REPO/$WHISPER_DIR/$BUILD_DIR/bin/whisper-cli"
if [[ -x "$REPO/$WHISPER_DIR/build-blas/bin/whisper-cli" ]]; then
    WHISPER_BIN="$REPO/$WHISPER_DIR/build-blas/bin/whisper-cli"
    info "whisper-cli already built"
elif [[ -x "$WHISPER_BIN" ]]; then
    info "whisper-cli already built"
else
    run cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${CMAKE_EXTRA[@]}"
    run cmake --build "$WHISPER_DIR/$BUILD_DIR" -j "$(nproc)" --target whisper-cli
fi
WHISPER_MODEL_FILE="$WHISPER_DIR/models/ggml-${WHISPER_MODEL}.bin"
if [[ -s "$WHISPER_MODEL_FILE" ]]; then
    info "$WHISPER_MODEL_FILE present"
else
    run bash "$WHISPER_DIR/models/download-ggml-model.sh" "$WHISPER_MODEL"
fi

# -- 4. voice -----------------------------------------------------------------

step "Speech synthesis (Piper voice)"
PIPER_VOICE="en_US-lessac-medium"
PIPER_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
ensure_dir vendor/piper
for ext in onnx onnx.json; do
    if [[ -s "vendor/piper/${PIPER_VOICE}.${ext}" ]]; then
        info "vendor/piper/${PIPER_VOICE}.${ext} present"
    else
        run curl -fL --retry 3 -o "vendor/piper/${PIPER_VOICE}.${ext}" "${PIPER_URL}/${PIPER_VOICE}.${ext}"
    fi
done

# Speaker recognition is optional: a failed download is a note, not an abort.
FETCH_SPEAKER='from homeai.speaker_admin import fetch_model; from homeai.config import SpeakerConfig; print(fetch_model(SpeakerConfig().model_path))'
if [[ $DRY_RUN -eq 1 ]]; then
    info "would download the speaker-recognition model (SHA256-checked)"
elif SPEAKER_OUT="$(.venv/bin/python -c "$FETCH_SPEAKER" 2>&1)"; then
    info "$SPEAKER_OUT"
else
    note "speaker-recognition model not downloaded (${SPEAKER_OUT##*$'\n'}); later: homeai-mode speaker fetch"
fi

# -- 5. ollama ------------------------------------------------------------------

step "Language model (Ollama)"
curl -fsS --max-time 5 http://127.0.0.1:11434/api/version >/dev/null \
    || die "Ollama is installed but not running. Start it (sudo systemctl start ollama) and re-run."
if ollama list | awk 'NR>1 {print $1}' | grep -x "${VOICE_BASE_MODEL}" >/dev/null; then
    info "${VOICE_BASE_MODEL} already pulled"
else
    run ollama pull "${VOICE_BASE_MODEL}"
fi
# Re-creating a loaded model unloads it, so only create when it differs.
CURRENT_CTX="$(ollama show llama31-voice --parameters 2>/dev/null | awk '$1=="num_ctx" {print $2}')"
if [[ "$CURRENT_CTX" == "$NUM_CTX" ]]; then
    info "llama31-voice exists with num_ctx ${NUM_CTX}"
else
    MODELFILE="$(mktemp)"
    trap 'rm -f "$ASSESS_ENV" "$MODELFILE"' EXIT
    sed "s/^PARAMETER num_ctx .*/PARAMETER num_ctx ${NUM_CTX}/" deploy/ollama/llama31-voice.Modelfile > "$MODELFILE"
    run ollama create llama31-voice -f "$MODELFILE"
fi
GUARD=/etc/systemd/system/ollama.service.d/homeai-vram-guard.conf
if [[ -f "$GUARD" ]] || systemctl show ollama -p Environment 2>/dev/null | grep OLLAMA_MAX_LOADED_MODELS=1 >/dev/null; then
    info "Ollama VRAM guard in place"
elif [[ $USE_SUDO -eq 1 ]]; then
    run sudo install -D -m 644 deploy/ollama-vram-guard.conf "$GUARD"
    run sudo systemctl daemon-reload
    run sudo systemctl restart ollama
else
    note "Ollama VRAM guard not installed (needs sudo). It stops Ollama stacking models until the machine runs out of memory. Re-run with --sudo, or: sudo install -D -m 644 deploy/ollama-vram-guard.conf $GUARD && sudo systemctl daemon-reload && sudo systemctl restart ollama"
fi

# -- 6. zeroclaw ------------------------------------------------------------------

step "ZeroClaw voice agent"
ZC_ARGS=(--config "$ZC_DIR/config.toml" --repo "$REPO" --num-ctx "$NUM_CTX" --agent "$AGENT")
[[ $DRY_RUN -eq 1 ]] && ZC_ARGS+=(--dry-run)
python3 -m homeai.install.zc_config "${ZC_ARGS[@]}" \
    || die "could not add the voice agent to $ZC_DIR/config.toml (reason above). Nothing there was changed."
WORKSPACE="$ZC_DIR/agents/$AGENT/workspace"
ensure_dir "$WORKSPACE" "$ZC_DIR/voice-scratch"
for persona in SOUL.md AGENTS.md; do
    if [[ ! -f "$WORKSPACE/$persona" ]]; then
        run install -m 644 "deploy/zeroclaw/$persona" "$WORKSPACE/$persona"
    elif cmp -s "deploy/zeroclaw/$persona" "$WORKSPACE/$persona"; then
        info "$persona up to date"
    else
        info "$WORKSPACE/$persona differs from deploy/zeroclaw/$persona; keeping yours"
    fi
done

# -- 7. env and service ---------------------------------------------------------

step "Configuration and service"
env_set_default HOMEAI_AGENT_NAME "$AGENT"
ZC_BIN="$(command -v zeroclaw)"
[[ "$ZC_BIN" == "$HOME/.cargo/bin/zeroclaw" ]] || env_set_default HOMEAI_ZEROCLAW_BIN "$ZC_BIN"
[[ "$WHISPER_BIN" == "$REPO/vendor/whisper.cpp/build-blas/bin/whisper-cli" ]] \
    || env_set_default HOMEAI_WHISPER_BIN "$WHISPER_BIN"
[[ "$WHISPER_MODEL" == "small.en" ]] \
    || env_set_default HOMEAI_WHISPER_MODEL "$REPO/$WHISPER_MODEL_FILE"
ensure_dir "$HOME/.local/share/homeai"

UNIT="$HOME/.config/systemd/user/homeai.service"
if [[ $INSTALL_SERVICE -eq 0 ]]; then
    info "skipping the service (--no-service); run: .venv/bin/python -m homeai.daemon"
elif [[ "$SYSTEMD_USER" != "1" ]]; then
    note "no systemd user session; run the assistant with: .venv/bin/python -m homeai.daemon"
else
    RENDERED="$(mktemp)"
    trap 'rm -f "$ASSESS_ENV" "${MODELFILE:-}" "$RENDERED"' EXIT
    sed -e "s#@REPO@#${REPO}#g" -e "s#@ZEROCLAW_DIR@#${ZC_DIR}#g" deploy/systemd/homeai.service.in > "$RENDERED"
    if [[ -f "$UNIT" ]] && cmp -s "$RENDERED" "$UNIT"; then
        info "service unit up to date"
    else
        [[ -f "$UNIT" ]] && run cp "$UNIT" "$UNIT.bak.$(date +%s)"
        run install -D -m 644 "$RENDERED" "$UNIT"
        run systemctl --user daemon-reload
    fi
    run systemctl --user enable homeai.service
fi

# -- 8. verify --------------------------------------------------------------------

if [[ $DRY_RUN -eq 1 ]]; then
    step "Verify (skipped in a dry run)"
elif [[ $VERIFY -eq 0 ]]; then
    step "Verify (skipped: --skip-verify)"
else
    step "Verifying"
    (set -a; [[ -f .env ]] && . ./.env; set +a; .venv/bin/python -m homeai.daemon --check) \
        || die "the assistant's startup checks failed (problems above)."
    info "asking the agent one question (loads the model; first time ~10-30s)..."
    # A question with a checkable answer: "reply with the word ready" was
    # once answered with the weather, which proves the pipeline but not sense.
    REPLY_TEXT="$("$ZC_BIN" agent --config-dir "$ZC_DIR" -a "$AGENT" -m "What is seven plus five? Answer in one short sentence." 2>/dev/null | tail -5 || true)"
    if [[ -z "${REPLY_TEXT// /}" ]]; then
        die "the ZeroClaw agent '$AGENT' gave no reply. Try: zeroclaw agent -a $AGENT -m hello"
    fi
    if ! printf '%s' "$REPLY_TEXT" | grep -Ei '\b(12|twelve)\b' >/dev/null; then
        note "the agent replied but did not answer 7+5 correctly; check the model with: zeroclaw agent -a $AGENT -m 'what is seven plus five?'"
    fi
    info "agent replied: $(printf '%s' "$REPLY_TEXT" | tr '\n' ' ' | cut -c1-80)"
    PROC="$(ollama ps | awk '$1 ~ /^llama31-voice/ {for (i=1;i<=NF;i++) if ($i ~ /GPU|CPU/) {print $(i-1), $i; exit}}')"
    info "ollama: llama31-voice on ${PROC:-unknown}"
    if [[ "$USES_GPU" == "1" && "$PROC" != *"100% GPU"* ]]; then
        note "the model is not fully on the GPU (${PROC:-not loaded}); replies will be slower. Check GPU drivers/ROCm and what else is using VRAM (ollama ps)."
    fi
    if [[ $INSTALL_SERVICE -eq 1 && "$SYSTEMD_USER" == "1" ]]; then
        run systemctl --user restart homeai.service
        sleep 5
        systemctl --user is-active --quiet homeai.service \
            || die "homeai.service did not stay up. See: journalctl --user -u homeai -n 50"
        info "homeai.service is running. Say \"Hey Jarvis\"."
    fi
fi

printf '\nDone%s.\n' "$([[ $DRY_RUN -eq 1 ]] && echo ' (dry run: nothing was changed)')"
for n in "${NOTES[@]}"; do printf '  - %s\n' "$n"; done
