# Home AI runbook

Day-to-day operation and fixes. The reasoning behind each decision is in
`ROUTING_GUIDE.md`.

## Install on a new machine

```bash
git clone <this repo> ~/src/homeai && cd ~/src/homeai
./install.sh --dry-run     # assess the machine and list every action
./install.sh --sudo        # install, including the Ollama VRAM guard
```

The installer aborts, listing every reason, if the machine cannot run a home
AI. Minimum requirements (from `homeai/install/assess.py`):

| | GPU tier | CPU-only tier (slow, unmeasured) |
|---|---|---|
| GPU VRAM | 6.5 GB (8K context) / 7.5 GB (16K) | — |
| RAM | 8 GB | 16 GB |
| CPU threads | 4 (8 for whisper small.en) | 8 |
| Disk | 15 GB free | 15 GB free |

You also need a microphone and a speaker, Python 3.12+, git, curl, cmake, a
C++ compiler, Ollama and ZeroClaw. The installer prints how to install
anything that is missing.

Re-running is safe. The installer only appends to an existing ZeroClaw config,
never overwrites a persona file or a `.env` value you changed, and backs up
anything it replaces as `*.bak.<epoch>`. On a new machine the voice agent is
named `jarvis`; on the original box it is `local` (`HOMEAI_AGENT_NAME` in
`.env`).

## Everyday commands

| Task | Command |
|---|---|
| Status | `systemctl --user status homeai` |
| Logs | `journalctl --user -u homeai -f` |
| Restart after a code change | `systemctl --user restart homeai` |
| Check config and dependencies | `set -a; . ./.env; set +a; .venv/bin/python -m homeai.daemon --check` |
| What is loaded on the GPU | `ollama ps` (expect `llama31-voice`, `100% GPU`, `Forever`) |
| Recent conversations | `tail ~/.local/share/homeai/transcript.jsonl` |
| Talk to the voice agent without a mic | `zeroclaw agent -a "$HOMEAI_AGENT_NAME" -m "..."` |

## Talking to Jarvis

- "Hey Jarvis, …" asks a question.
- Saying "Hey Jarvis" while it is speaking interrupts it. Then "stop",
  "quiet" or "I'm not talking to you" ends the turn without a lookup.
- If a reply ends with a question, just answer; no wake word is needed.
- Long answers stop at about 110 words and ask "Want me to keep going?".
  "Yes" or "go on" continues; "no" or silence drops it.

## Tuning (`.env`, then restart)

| Variable | Default | Effect |
|---|---|---|
| `HOMEAI_SPOKEN_BUDGET_WORDS` | 110 | Words before "keep going?"; 0 disables |
| `HOMEAI_AGENT_STYLE_HINT` | built in | Guidance sent with every request; `off` disables |
| `HOMEAI_WAKE_THRESHOLD` | 0.5 | Raise it if Jarvis wakes by mistake |
| `HOMEAI_AGENT_NAME` | local | ZeroClaw agent used for voice |
| `HOMEAI_AGENT_TIMEOUT` | 45 | Seconds before giving up on the agent |

The persona lives in `~/.zeroclaw/agents/<agent>/workspace/SOUL.md`. The
shipped copy is `deploy/zeroclaw/SOUL.md`. After editing it, measure with
`tools/bench_conversation.py`; don't judge from one conversation.

## When something goes wrong

| Symptom | Check | Usual fix |
|---|---|---|
| Never wakes | Logs show `wake detector`? The right mic? | `HOMEAI_INPUT_DEVICE`, or lower the wake threshold |
| Wakes, then "Something went wrong" | `journalctl --user -u homeai -n 50` | `ollama ps`: the model isn't loaded. `ollama run llama31-voice ""` |
| Slow replies (over 5 s) | `ollama ps` shows a CPU % | Something else is using the VRAM (the coder's llama-server); see below |
| Repeats itself or recites lists | `tools/inspect_zc_memory.py`, `tools/probe_agent_prompt.py` | Make sure `memory_recall` is not in the voice profile's `allowed_tools` |
| Box froze, out of memory | `systemctl show ollama -p Environment` | Install `deploy/ollama-vram-guard.conf` (`OLLAMA_MAX_LOADED_MODELS=1`) |
| Service won't start: preflight | `.venv/bin/python -m homeai.daemon --check` | Fix each listed problem |

## Coding mode (the coder gets the whole GPU)

The voice model (7.0 GB) and qwen3-coder-30b (about 19.7 GB) together exceed
the 21.4 GB card. When they share it, the coder runs at about 25 tok/s.

```bash
homeai-mode coding    # Jarvis off, coder at ~131 tok/s (measured), ~7 s to switch
zc                    # code with ZeroClaw's builder agent, or use `hermes`
homeai-mode voice     # Jarvis back, ~9 s to switch
homeai-mode status
```

- Both `zc` (ZeroClaw builder) and Hermes use the same llama-server, so both
  get the speed-up.
- Coding mode defaults to a q4_0 KV cache with every layer on the GPU.
  `homeai-mode coding --kv q8_0 --cpu-moe 6` gives a higher-fidelity cache at
  about 80 tok/s.
- `--ctx N` changes the context window (default 65536). Larger contexts are
  unmeasured and may need `--cpu-moe` to fit.
- If the coder fails to start, the switch rolls back to voice mode.
- Coding mode does not survive a reboot. The machine always comes back in
  voice mode.

Tool-call limits for the builder agent (`[runtime_profiles.heavy_duty]` in
`~/.zeroclaw/config.toml`):
- 100 tool calls per request.
- No hourly action cap. ZeroClaw's default was 20 per hour, which stopped
  long tasks with "Rate limit exceeded".
- 600 s shell timeout. The default was 60 s.

Verify with `tools/probe_tool_cap.py --steps 40`. Hermes has its own limit:
500 turns.

## False wakes ("it answered and nobody said Jarvis")

A wake word alone isn't enough. Whisper must also hear "Jarvis" (or a close
mishearing such as Jervis or Jarvan) in the 2 s before the trigger plus the
utterance. Otherwise Jarvis stays silent and logs:

    unverified wake (score 0.937): Whisper heard "...", no wake word - ignoring

```bash
journalctl --user -u homeai | grep -E "wake word detected|unverified|heard:"
```
- `transcript.jsonl` verdicts include `unverified`, `wake-only`,
  `pleasantry` and `closing`.
- A thank-you ends a conversation; it doesn't start one. Within 5 minutes
  of an exchange, "thank you" or "bye" goes to the model with a
  one-sentence hint (`daemon.CLOSING_HINT`). The reply is then cut to its
  first sentence (`dialogue.one_sentence`) and never opens a follow-up
  window. If the reply is empty, fails, or is only a question, the canned
  "You're welcome." / "Bye for now." is used. A thank-you out of nowhere,
  "okay" on its own, or Whisper noise ("thanks for watching") gets silence.
- Replies that narrate the model's reasoning ("no need to call a tool
  function…") are discarded and retried.
- To turn the check off: `HOMEAI_WAKE_VERIFY=0` in `.env`. Pre-roll
  length: `HOMEAI_WAKE_PREROLL_S` (default 2.0).
- If real wakes are rejected, add the misheard word to
  `tests/test_wake_verify.py` and check it with
  `tools/wake_word_similarity.py WORD`.

## Changing Jarvis's voice

```bash
.venv/bin/python tools/piper_voice_samples.py --play   # hear every installed voice
```
Set `HOMEAI_PIPER_VOICE=<name>` in `.env`, then run
`systemctl --user restart homeai`. Voices are found in `vendor/piper/` or
`vendor/piper/voices/`. Current voice: `en_GB-alba-medium` (chosen
2026-10-02). Every installed voice starts speaking in about 0.6 s, so choose
by ear.

## Who is speaking (speaker recognition)

Off by default. Jarvis learns voices and then:
- tells the model who is speaking, so it can use your name now and then;
- keeps a separate conversation window per recognised person (an
  unrecognised turn stays in the current window, so a miss costs nothing);
- **ignores follow-ups (no wake word) from a voice that is clearly not the
  person it is talking to**, so family chatter is not taken as your answer.

Anyone, known or not, still gets full service. This is courtesy, not
security: a recording of your voice will pass.

```bash
homeai-mode speaker on        # downloads the model if needed, restarts homeai
homeai-mode speaker           # on/off, thresholds, who is enrolled
homeai-mode speaker off
homeai-mode speaker remove Adam   # takes effect without a restart
```

By voice (best: same mic and distance as everyday use):
- "Hey Jarvis, remember my voice as Adam." ("is Adam" or just "Adam" also
  work; with no name heard, Jarvis asks for one.) Then talk about anything for
  about ten seconds; Jarvis asks for more if it needs it and says "Got it"
  when done. Saying "stop" cancels. Re-enrolling replaces the old profile.
- "Hey Jarvis, who am I?" says who it thinks you are (no model call).
- "Hey Jarvis, forget my voice." removes the voice it recognises.

Checking how it does:
```bash
journalctl --user -u homeai | grep -E "speaker:|another voice|enroll"
```
Each line shows the best match and its score, e.g. `speaker: Adam (0.62,
margin 0.40)` or `unknown (best Adam 0.31: below threshold)`; the score is
also saved as `speaker_score` in transcript.jsonl. Tuning in `.env`:
`HOMEAI_SPEAKER_THRESHOLD` (0.45: score needed to name someone),
`HOMEAI_SPEAKER_MARGIN` (0.1: lead over the next-best voice),
`HOMEAI_SPEAKER_DIFFERENT_BELOW` (0.25: a follow-up scoring below this
against the person in the conversation is ignored),
`HOMEAI_SPEAKER_FOLLOWUP_GATE=0` turns that gate off.

Voices are stored in `~/.config/homeai/speakers.json` (mode 600). The model
is `vendor/speaker/wespeaker_en_voxceleb_resnet34_LM.onnx` (SHA256-checked,
~40 ms per utterance on the CPU).

## What ZeroClaw remembers (privacy)

ZeroClaw saves every request to every agent, including speech Jarvis
overheard, in `~/.zeroclaw/data/memory/brain.db`. These are `conversation`
rows, kept for 30 days by default. Jarvis cannot read them back, but anyone
with the disk can.

```bash
homeai-mode memory                  # saving on/off, retention, rows, disk used
homeai-mode memory off              # stop saving (existing rows stay)
homeai-mode memory on
homeai-mode memory retention 7      # days; 0 = keep forever
homeai-mode memory purge            # delete saved conversations (asks first; --yes to skip)
```

- `auto_save` is global: turning it off also affects `zc` and other agents.
- Purge deletes only `conversation` rows; notes saved on purpose (`core`,
  `daily`) are kept. Purge then VACUUMs the database, so the deleted text is
  really gone from disk, not just unlinked.
- Config changes restart `zeroclaw.service` so its retention sweep sees them.
- Test without touching live data: `tools/test_memory_privacy_sandbox.sh`
  (runs against a copy of `~/.zeroclaw`).

## Tests

```bash
.venv/bin/python -m pytest -p no:warnings -o addopts="" -q   # unit tests
tools/mutation/check_mutants.sh                              # are the tests real?
tools/test_installer_sandbox.sh                              # fresh install end to end (~5 min)
.venv/bin/python tools/probe_voice_tools.py                  # voice agent cannot touch the machine
```
