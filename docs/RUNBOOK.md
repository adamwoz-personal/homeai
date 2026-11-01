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

## Long-term memory (what Jarvis remembers about you)

This is separate from ZeroClaw's memory below. Jarvis keeps each exchange with
a **recognised** speaker in Postgres (`homeai` database), and adds relevant
memories to its prompt before answering. It never records home commands,
closings, unknown speakers, or test voices (`HOMEAI_LTM_IGNORE_SPEAKERS`,
default `ryan,amy,joe,alba,hfc`). Secrets such as card numbers, API keys,
and SSNs are redacted before storage.

```bash
bin/homeai-memory stats                     # counts, newest, size
bin/homeai-memory list -n 10                # recent episodes (yours; --owner X as admin)
bin/homeai-memory search "the snake"        # what recall would find
bin/homeai-memory note "Allergic to cats" --profile   # always remembered (--pin, --for NAME)
bin/homeai-memory forget episode 42         # delete one (audit keeps no text)
bin/homeai-memory export > me.json
bin/homeai-memory add-person sue [--child]  # name must match the enrolled speaker
bin/homeai-memory backfill --dry-run        # import past transcript turns
bin/homeai-memory facts [--kind preference] # notes drawn from conversations
bin/homeai-memory profile                   # the short profile built from them
bin/homeai-memory extract [--all]           # draw notes now (admin; normally automatic)
bin/homeai-memory reflect                   # rebuild your profile now
bin/homeai-memory jobs                      # background queue (admin)
```

**Notes and profile (automatic).** About 10 minutes after a conversation
goes quiet, a small CPU model (`qwen3:4b-instruct` in Ollama, never the GPU)
reads the new exchanges and keeps a few notes: facts, preferences, plans,
opinions. It never keeps commands or anything Jarvis said that you didn't
(a note using a word only Jarvis used is thrown away). Similar notes are
merged; corrections replace the old note. Half an hour later a short profile
is rebuilt from the notes. Children get no opinion/thought notes. Quality
check on real and made-up conversations:
`.venv/bin/python tools/memory/extract_eval.py [--fixture]` (writes a review
sheet in `~/homeai-bench/`).

**Say it to Jarvis** (recognised voices only):

| Say | Does |
|---|---|
| "remember that I'm allergic to cats" / "always remember …" | keeps it (always: pinned) |
| "what do you remember about me" / "… about the snake" | lists up to 5 |
| "that's wrong, it's Noodle" (right after a list) | corrects the closest one |
| "forget that" | deletes the last exchange (no question asked) |
| "forget what I said about the snake" / "forget everything about me" | asks first, then deletes |
| "stop remembering" / "start remembering" | pauses / resumes for you |

- Off switch: `HOMEAI_LTM=0` in `.env`, then restart. Other settings:
  `HOMEAI_LTM_DSN`, `HOMEAI_LTM_MIN_SIM` (0.60),
  `HOMEAI_LTM_TIMEOUT_MS` (150), `HOMEAI_LTM_BUDGET` (600 tokens).
  Notes: `HOMEAI_LTM_EXTRACT` (1), `HOMEAI_LTM_HELPER_URL` (Ollama),
  `HOMEAI_LTM_HELPER_MODEL`, `HOMEAI_LTM_HELPER_THREADS` (6),
  `HOMEAI_LTM_EXTRACT_IDLE_S` (600), `HOMEAI_LTM_EXTRACT_BATCH` (20),
  `HOMEAI_LTM_REFLECT_DELAY_S` (1800).
- Forgetting removes the exchange, notes drawn only from it, profile lines
  built from them, and unused topic names (checked by
  `tests/test_memory_forget_dump.py` against a full `pg_dump`). Deleted rows
  stay on disk until autovacuum; ZeroClaw's own memory is separate (below).
- Full dump (row security applies even to the owner):
  `PGOPTIONS='-c app.scope=admin' pg_dump --enable-row-security homeai`.
- If Postgres is down, Jarvis answers without memories, logs one warning
  every 10 minutes, and reconnects on its own.
- Log line on start: `long-term memory ready`.
- Speed check (scratch database, ~4 min): `.venv/bin/python tools/memory/recall_bench.py`.
- One-time setup already done: Postgres 18 + pgvector, `adam` has
  CREATEROLE/CREATEDB, and `vector` is in `template1`.

**Backups.** A user timer (`homeai-memory-backup.timer`, 03:30 daily) dumps
the database to `~/.local/share/homeai/backups/` (mode 0700, last 14 kept),
then restores it into a scratch database and checks row counts, schema
version and forced row security. A failed check fails the unit.

```bash
bin/homeai-memory backup [--keep N] [--no-check]   # dump now (+ restore check)
bin/homeai-memory restore-check [FILE]             # check newest (or FILE) again
systemctl --user list-timers homeai-memory-backup.timer
journalctl --user -u homeai-memory-backup -n 20
```

Settings: `HOMEAI_LTM_BACKUP_DIR`, `HOMEAI_LTM_BACKUP_KEEP` (14). Disaster
recovery (stop `homeai` first). Roles are cluster-wide, not in the dump: on a
brand-new Postgres, create them once with `createdb scratch &&
HOMEAI_LTM_DSN=dbname=scratch bin/homeai-memory migrate && dropdb scratch`.

```bash
dropdb homeai && createdb homeai
PGOPTIONS='-c app.scope=admin' pg_restore --no-comments -d homeai \
    ~/.local/share/homeai/backups/memory-<stamp>.dump
bin/homeai-memory migrate     # applies anything newer than the dump
```

**Memory tools for coding agents (MCP).** `bin/homeai-memory-mcp --scope
admin|person --user NAME --client NAME` serves `memory_search`, `memory_get`,
`memory_add`, `memory_update`, `memory_supersede`, `memory_delete`,
`memory_topics`, `memory_health` over stdio, as the `homeai_mcp` role.
Registered for ZeroClaw's `builder` (bundle `memory-admin`, client
`builder`) and Copilot CLI (`~/.copilot/mcp-config.json`, client
`copilot`), both admin scope as `adam`.

- Never give it to the voice agent. `python3 -m homeai.install.zc_config
  --check` fails if any voice agent can reach the `memory` or `ha` server;
  the installer refuses such a config too.
- Person scope: reads own + household, writes own only. Admin scope needs
  `is_admin` in `memory.people`. The scope is a per-transaction setting any
  local program could set: it stops agent mistakes and injected arguments,
  not local code.
- Agents can't write `instruction`, `profile` or `summary` facts; theirs are
  `source=mcp`, `trust=household`. A near-duplicate (similarity ≥ 0.92) is
  not added; 0.65–0.92 matches come back as `similar` so the agent decides
  whether to supersede. Deletes record a reason word (`wrong`, `outdated`,
  `duplicate`, `requested`, `dead_end`) and the client, never text.
- Checks: `.venv/bin/python tools/memory/mcp_probe.py` (every tool over
  stdio on the real database, cleans up) and
  `tools/memory/builder_memory_check.sh` (the builder agent uses the tools;
  verified from the audit log).

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

## Home Assistant container

Device backend for Jarvis (plan: `plans/HOME_ASSISTANT_PLAN.md`). Docker
container `homeassistant`, host networking, config in `~/homeassistant-config`,
restarts on boot.

- Status / start / stop / logs: `tools/ha/ha_container.sh status|start|stop|logs`
- Upgrade to the latest stable: `tools/ha/ha_container.sh update` (config is kept)
- Web UI: http://192.168.1.242:8123 (LAN) or http://127.0.0.1:8123
- Log noise is silenced by `tools/ha/ha_quiet_logs.py` (idempotent; `--dry-run`,
  `--scan --hours N`). It disables HA's Bluetooth entry (we don't use it) and
  sets `aioamazondevices` logging to error. Re-run it after rebuilding the
  config. Backups are `*.homeai-bak` in `~/homeassistant-config`.

## Home Assistant MCP for coding

Lets Copilot CLI and ZeroClaw's `builder` agent inspect/control HA via HA's own
MCP server (Assist API, exposed entities only). Never attach it to the voice
agent, and keep the garage out of Assist exposure.

1. HA UI: Settings > Devices & services > Add > "Model Context Protocol
   Server". By default it needs an admin user; turn off "Require admin" in its
   options to use a non-admin one.
2. Create a long-lived token for that user and save it in
   `~/.config/homeai/ha-coding.env` (mode 600) as `HA_URL=` / `HA_TOKEN=`
   (separate from the voice token; `HOMEAI_HA_MCP_ENV` overrides the path).
   Currently the same `jarvis` user/token as `ha.env` with "Require admin"
   off; split them later by creating a separate user and token.
3. Check: `.venv/bin/python tools/ha/ha_mcp_check.py`.
4. Clients run `bin/homeai-ha-mcp` (stdio proxy, `homeai/ha_mcp_proxy.py`).

## Home control (Jarvis -> Home Assistant)

- Credentials: `~/.config/homeai/ha.env` (mode 600) with `HA_URL=` and
  `HA_TOKEN=` (long-lived token of the non-admin `jarvis` HA user). Env vars
  override; `HOMEAI_HA_ENV` points elsewhere.
- `HOMEAI_HOME=0` disables home control; `HOMEAI_HOME_FASTPATH=0` sends every
  home request through the model instead; `HOMEAI_HA_DEFAULT_ECHO` (default
  `kitchen`) picks the Echo for timers and music.
- Ring doorbell (`homeai/doorbell.py`): a ring is announced on the Echos
  ("Someone is at the front door."). Too chatty? `HOMEAI_DOORBELL=0` turns it
  off; `HOMEAI_DOORBELL_QUIET_HOURS=22:00-07:00` silences it overnight;
  `HOMEAI_DOORBELL_WHERE=kitchen` limits it to one Echo;
  `HOMEAI_DOORBELL_MOTION=1` also announces motion (off by default). Ask
  "who's at the door" / "how's the doorbell battery". Live check:
  `.venv/bin/python tools/ha/test_doorbell_live.py` (`--announce` speaks).
  Set the variables in the service's environment, then restart homeai.
- Startup log line: `home control: Home Assistant at ... (fast path on)`.
  Each action logs `home <kind> {...} -> <reply>` and `HA call ...`.
- Set up from scratch: `./install.sh --home-assistant` (add `--sudo` to let it
  install Docker). It checks Docker, starts the container, walks through
  onboarding, validates the token you paste (hidden, never printed or logged;
  `HOMEAI_HA_TOKEN` supplies it non-interactively) and writes `ha.env`
  (mode 600). It lists the browser-only steps: Hue button, Alexa login with
  the authenticator code, Ring, the phone app. Safe to re-run.
- `--check` failures say what to do: HA down -> `tools/ha/ha_container.sh
  status|start`; token rejected -> new long-lived token; package missing ->
  re-run `install.sh`; no lights/Alexa devices -> that integration isn't added.
- Routines (Jarvis does them; the `jarvis` HA user can't create HA
  automations): "goodnight routine" / "I'm going to bed" turns every light off.
  A bare "good night" stays a farewell. With the Home Assistant app on a phone,
  rings are also pushed to it (`HOMEAI_DOORBELL_PHONE=0` stops that); the push
  is not silenced by quiet hours. A garage-left-open alert waits on
  `garage-hardware`.
- Quick connection check (also run by `install.sh`):
  `.venv/bin/python -m homeai.ha_client --check` (exit 0 working, 1 not set
  up, 2 configured but broken).
- Live check (read-only): `.venv/bin/python tools/ha/test_home_live.py`;
  `--act` toggles a Hue room and sets/cancels a timer, restoring both;
  `--announce` speaks on every Echo.
- Echo list / manual commands: `tools/ha/ha_alexa_probe.py`
  (`--command kitchen "cancel all timers"` stops a runaway timer).
- Timer cancel behaviour: `tools/ha/timer_cancel_experiment.py`.

