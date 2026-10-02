#!/usr/bin/env bash
# Speak TEXT through the speakers in a Piper voice, so the live daemon hears it
# through the real microphone, then show what it logged. For end-to-end checks
# without a human:  tools/say_to_jarvis.sh "Hey Jarvis. What is two plus two?"
#   VOICE=en_US-ryan-high WAIT=25 tools/say_to_jarvis.sh "..."
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
TEXT=${1:?usage: say_to_jarvis.sh TEXT}
VOICE=${VOICE:-en_US-joe-medium}   # not Jarvis's own voice, to keep logs readable
WAIT=${WAIT:-20}
ONNX=$(ls "$REPO"/vendor/piper/{,voices/}"$VOICE".onnx 2>/dev/null | head -1 || true)
[ -n "$ONNX" ] || { echo "voice $VOICE not found" >&2; exit 1; }
WAV=$(mktemp --suffix .wav); trap 'rm -f "$WAV"' EXIT
"$REPO/.venv/bin/piper" --model "$ONNX" --output_file "$WAV" <<<"$TEXT" 2>/dev/null
DEVICE=$(cd "$REPO" && .venv/bin/python -c 'from homeai.config import AudioConfig; print(AudioConfig().output_device)')
since=$(date '+%Y-%m-%d %H:%M:%S')
aplay -q -D "$DEVICE" "$WAV"
sleep "$WAIT"
journalctl --user -u homeai --since "$since" --no-pager -o cat | grep -E "wake word|heard:|fragment|dismissal|perceived lag|discarding" || echo "(daemon logged nothing)"
