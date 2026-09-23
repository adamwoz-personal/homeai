# ROUTING_GUIDE.md

**Purpose:** which work goes to ZeroClaw (local `qwen3-coder-30b`, free) and which stays
with Claude (paid). Updated as evidence accumulates. Every rule here comes from an
observed outcome on this box, not from theory.

Last updated: 2026-09-14

---

## The hard limits of ZeroClaw on this host

| Limit | Value | Evidence | Consequence |
|---|---|---|---|
| **Tool iterations per message** | **10** | TASK-01 aborted mid-build: *"Turn stopped: reached maximum tool iterations (10)"* | Any task needing >10 tool calls **will** fail halfway. Decompose or do it yourself. |
| Malformed tool-call markup | ~10–20% of tool-calling turns | 1/12 on raw curl; 2/10 clean history | Must verify artifacts; never trust the summary |
| Session poisoning | 100% once history is contaminated | 10/10 after poisoning vs 2/10 clean | **Always** pass `--session-state-file` with a unique path per task |
| Reports success without checking | Observed | Wrote a 371-byte stub, claimed the plan was complete | Acceptance criteria must be explicit *and* independently verified |
| LLM slot contention | `--parallel 1` | Config | A long ZeroClaw task blocks every other LLM consumer |

---

## Decision rule

```
Is the task mechanical, bounded, and verifiable by a file check?
├── YES → Can it finish in under ~8 tool calls?
│         ├── YES → DELEGATE to ZeroClaw
│         └── NO  → Split into sub-tasks of <8 calls, then delegate each
└── NO  → Does failure have a security or correctness blast radius?
          ├── YES → Claude. Always.
          └── NO  → Claude if it needs diagnosis; ZeroClaw if it needs typing
```

---

## Proven good ZeroClaw tasks

| Task | Result |
|---|---|
| TASK-02: download two Piper files to a path | **Success.** Correct files, correct sizes, honest report |
| Smoke test: write a file with exact content | **Success** |

Characteristics that made these work:
- Single clear goal, no branching
- Under 10 tool calls
- Acceptance criteria stated as file existence + size
- Explicitly told *"do not claim success without checking"*

## Proven bad ZeroClaw tasks

| Task | Failure |
|---|---|
| TASK-01: clone + configure + build + download model | Hit the 10-iteration cap after cloning. Reported honestly, but wasted a cycle. |
| Writing `homeai.plan.initial` | Produced a 371-byte stub, silently, and claimed success |
| Both original hardware plans | Invented PulseAudio and `hw:0,3`; never inspected the machine |

Characteristics that made these fail:
- Multi-stage with a long-running step in the middle
- Required inspecting reality before writing
- Success was a judgement call, not a file check

---

## Rules that follow

1. **Always pass `--session-state-file /tmp/zc-<task>.json`** with a unique name.
   Prevents poisoning carry-over between tasks.
2. **Always verify the artifact yourself.** `ls -l`, byte size, and run it.
   ZeroClaw's dominant failure is a confident report over an empty result.
3. **Budget 8 tool calls per delegated message.** The cap is 10; leave headroom
   for the model's own reads and re-reads.
4. **Long builds: run them directly.** A `cmake --build -j24` needs no judgement.
   Delegating it adds a failure mode and saves nothing.
5. **Write the sub-plan to a file, then tell ZeroClaw to read and execute it.**
   Keeps the prompt short and gives a durable record of what was asked.
6. **State acceptance criteria as checkable facts** — "file X exists and exceeds
   40 MB" — never "the component works".
7. **Never delegate security-relevant code.** The safety filters and agent
   privilege profile stay with Claude.

---

## Task routing for this project

| Component | Owner | Why |
|---|---|---|
| `safety.py` | **Claude** | Guards a privileged agent against arbitrary voice input |
| `agent_client.py` | **Claude** | Poisoning guard + failure semantics; subtle |
| `audio.py` ring buffer | **Claude** | Wrap-around logic is easy to get subtly wrong |
| `config.py` | **Claude** | Hardware facts had to be measured, not guessed |
| Unit tests | **Claude** | Tests encode intent; a wrong test is worse than none |
| whisper.cpp build | Claude (direct) | Long single command; delegation adds risk only |
| Model/voice downloads | **ZeroClaw** | Bounded, file-checkable. Proven to work |
| Dependency installs | **ZeroClaw** | Bounded and verifiable |
| `daemon.py` wiring | **Claude** | Concurrency + must-never-die semantics |
| Documentation updates | Either | Low blast radius |

---

## Lessons beyond ZeroClaw

Recorded because they cost real time this session.

1. **A non-breaking space (U+00A0) in a pasted command** produced
   `Error: Invalid operation install ffmpeg`. Bash does not word-split on NBSP,
   so `install<NBSP>ffmpeg` became one argument. When a trivial command fails
   inexplicably, check the bytes before blaming the tool.
2. **`enable-linger` does not create a seat session.** Audio device access came
   from a logind ACL tied to an active tty. Would have worked in every test and
   failed on the first unattended reboot. Fixed with `usermod -aG audio`.
3. **Verify claims against the machine, not the document.** Both inherited plans
   named the wrong audio stack and the wrong devices.
4. **A hung tool call is worse than a failed one.** Hermes ran `speaker-test`
   with no `-l 1` and blocked its own agent for 26 minutes. Always bound
   external commands with a timeout or an explicit loop count.
5. **`uv pip install silero-vad` pulls a CUDA PyTorch** — 5.7 GB for a 2 MB
   model, on an AMD box. Check what a convenience package drags in.

---

# Session 2 — lessons learned

## SECURITY: the gateway webhook does not enforce risk profiles

**This is the most important finding in this file.**

Measured, reproducibly:

| Path | `date` (not in allowlist) | Latency |
|---|---|---|
| `POST /webhook` | **executed and returned output** | ~5.2 s |
| `zeroclaw agent -a local -m ...` | **refused** | ~0.5 s |

The agent bound to the gateway had `level = "readonly"` and an
`allowed_commands` list that excluded `date`. The CLI honoured it. The webhook
did not.

Things that did **not** change the webhook's behaviour:
- adding `"agent": "voice"` to the request body (field is ignored)
- setting `[acp] default_agent`
- rebinding `[agents.local]` to the restricted profile

Consequence: **the voice path must not use the gateway webhook.** A voice
waveform carries no authentication — a television, a radio, or a guest can
issue commands. `homeai` therefore talks to ZeroClaw through
`homeai/agent_cli.py` (subprocess), which is both enforcing *and* ~10x faster.

`HOMEAI_AGENT_TRANSPORT=http` still selects the old path, but it logs a loud
warning. Do not use it for voice.

## Restricted profiles are a large latency win, not a cost

The restricted agent loads ~3 tools instead of 62. That alone took a steady
reply from ~4.2 s to ~0.5 s. Security and speed pointed the same direction
here, which is rare — take it.

Cold start for a given agent is still ~25 s, so the daemon warms it at startup
(now 0.3 s, since the warm-up itself is cheap on a small toolset).

## ZeroClaw agent/profile mechanics

- Valid `AutonomyLevel`: `readonly`, `supervised`, `full`. There is no
  `strict`/`minimal` level for profiles despite those strings appearing
  elsewhere in the schema.
- Per-agent persona files live at
  `~/.zeroclaw/agents/<name>/workspace/{SOUL.md,AGENTS.md}` — **not** in
  `~/.zeroclaw/`. `zeroclaw doctor` reports missing ones per agent, which is a
  quick way to confirm an agent is actually registered.
- The gateway appears pinned to the agent named `local`. Rather than fight it,
  `local` is now the **restricted** voice agent and a new **`builder`** agent
  holds the privileged profile.
- **Delegation now uses `zeroclaw agent -a builder`, not `-a local`.**
- `zeroclaw gateway restart` can hang indefinitely. Use
  `systemctl --user restart zeroclaw.service` instead, and check for orphaned
  `zeroclaw gateway restart` processes if behaviour looks stale.
- ZeroClaw preserves hand-written config now that it parses; verify with
  `md5sum` before/after a `doctor` run if you suspect a silent reset.

## SOUL.md is an effective output-format control

Telling the voice agent to write numbers as spoken words worked first time:
> "Mount Everest is eight thousand eight hundred forty-eight meters tall."

Markdown suppression matters more than it sounds — Piper reads `**` and
backticks aloud.

## Delegation scorecard

| Task | Outcome |
|---|---|
| TASK-01 whisper build | Failed — exceeded the 10 tool-call cap |
| TASK-02 Piper download | Clean success |
| TASK-03 README | Clean success — all 6 headings, touched no `.py` |

Pattern holds: **bounded, mechanical, verifiable tasks succeed.** TASK-03's
acceptance criteria were checkable in one command, which is why it worked.

ZeroClaw still over-claims. It reported the README "is 40 lines, which exceeds
the minimum requirement of 40 lines". It was exactly 40. Always verify the
artifact yourself — `wc -l`, `grep` for each required heading, and `stat` to
confirm it touched nothing it shouldn't have.

---

# Session 3 — lessons learned

## Don't pay an LLM to run `wget`

TASK-05 (download Whisper models) was dispatched to ZeroClaw. It burned tool
calls failing to `cd`, and downloading needs no intelligence whatsoever. Doing
it in plain bash took one command and cost nothing.

**Rule: delegate judgement, not mechanics.** If a task is a fixed sequence of
shell commands with no decisions in it, just run it. Reserve the local agent
for work that involves reading, writing, or deciding.

TASK-06 (write unit tests for `transcript.py`) was the right shape: it needed
reading code and writing prose-like output. It produced 7 correct tests that
passed on the first run — but it could not run pytest itself under the
`builder` profile, and said so. **It reported the limitation honestly rather
than claiming success**, which is a notable improvement over the README
over-claim. Verify anyway.

## Benchmarks lie until the metric is right, twice over

Benchmarking Whisper models produced three different answers:

| version of the test | verdict |
|---|---|
| naive WER | medium.en best (4.1%), base.en 5.5% |
| + number normalisation | **everything ties at 0.0%** |
| + reverb and noise | **small.en best**; base.en fails |

1. The naive metric counted `"twenty five"` → `"25"` as an error. Whisper was
   *correct*; the metric was wrong. Fixing it erased every apparent difference.
2. Clean Piper audio is too easy — all models scored perfectly, so the test had
   no discriminating power at all. Only after adding reverb and brown noise
   (`--degrade`) did real differences appear.

The degraded run reproduced the exact live failure class: `base.en` heard
"Mount Everest" as "**melt** Everest", the same shape as the real
"What **if** the capital of France". That is the evidence that justified
switching the default to `small.en` — 0.0% WER for +0.38s.

**Lesson: if a benchmark shows no difference, suspect the benchmark before
concluding there is no difference.**

## Resetting a detector is not the same as making it deaf

`openWakeWord.reset()` was already being called after each capture, yet a
second wake still fired 0.245s later and then recorded silence for 25 seconds,
stalling the pipeline. The model retains ~1s of internal audio context that
`reset()` does not clear. The fix is a **refractory window** (`refractory_s`,
default 1.5s) during which wakes are ignored outright.

## systemd user services cannot drop capabilities

`ProtectKernelTunables`, `ProtectKernelModules`, and `ProtectControlGroups`
all fail a **user** unit with `218/CAPABILITIES`. Also `StartLimitIntervalSec`
and `StartLimitBurst` belong in `[Unit]`; in `[Service]` systemd ignores them
silently and the restart limiter never engages.

---

# How the bugs were actually found

The mechanical lessons above are specific to this box. These are the habits
that produced them, and they generalise.

## Write the adversarial test even when you are sure the code is right

`safety.py` was written to catch leaked tool-call markup, and it looked
correct. Writing a test for the *bare JSON* case — `{"name": "shell_tool",
"arguments": {"command": "rm -rf /"}}` — proved it was not: every one of the
six patterns was tag-based, so the model's **dominant** failure mode passed
straight through and would have been spoken aloud.

The hole was found by writing a test for a case I expected to pass. That is
the entire value of the exercise. A test that only confirms what you already
believe has bought you nothing.

Corollary: when hardening a detector, immediately add **false-positive** tests
too ("the *function* of the heart", "two *arguments* about salt"). Tightening
a security check until it breaks ordinary use is a different bug, not a fix.

## Never conclude from a single cold measurement

The restricted voice agent first measured **24.1s** — apparently a disaster
and nearly grounds for abandoning the design. Warm, the same query took
**0.50s**. The first call was paying a one-off model/agent load.

Always take at least two measurements and say which is which. Cold-start cost
is a real number, but it is a *different* number from steady-state latency,
and conflating them leads to discarding the correct design.

## When results stop making sense, look for a stuck process

Several webhook experiments returned contradictory results. The cause was an
orphaned `zeroclaw gateway restart` that had hung, so the config under test
was never actually loaded and I was measuring a stale daemon.

Symptom to watch for: **an experiment whose result does not change when it
should.** Before theorising, check `pgrep -af` for a previous step that never
exited, and prefer `systemctl --user restart` over the tool's own restart verb.

## Change one thing at a time — a rule I broke and paid for

Early on, four config settings were changed together to fix the tool-call
problem. It worked, and **it is still unknown which one mattered**. That
ignorance is now permanent unless someone re-tests from scratch, and it means
`native_tools` may well be a no-op for llamacpp that we are carrying forever.

The later webhook work was done one variable at a time, which is exactly why
the conclusion there is trustworthy.

## Verify the premise before acting on the request

A request to "loosen the Hermes config, it's too restrictive" turned out to be
aimed at the wrong component: testing showed Hermes was permissive and
**ZeroClaw** was the restrictive one. Acting directly would have loosened
security on the wrong system and not fixed the problem.

Cheap check, large payoff: reproduce the reported symptom before changing
anything.

## Duck-typed interfaces drift silently

`CliAgentClient` was written with `health_ok()` while the daemon called
`health()`. Nothing failed at import; it failed at service start, in
production, with a `AttributeError`. Two interchangeable implementations need
an explicit parity test asserting they expose the same methods — now
`test_cli_and_http_clients_expose_the_same_interface`.

## Prefer the log over the theory

The "noticeable delay" had an obvious plausible explanation (the LLM is slow).
The journal showed something else entirely: a spurious second wake and a
25-second dead capture. No amount of reasoning about model latency would have
found that.

When the daemon could not answer a question about its own behaviour, the right
move was to **add instrumentation first** (per-stage timings, transcript log)
rather than guess harder.

---

# Open questions, honestly recorded

These are unresolved. Do not assume either way.

1. **Does `allowed_commands = []` mean "deny all" or "unset, allow all"?**
   Suspected the latter, never proven — the experiment was contaminated by the
   stuck-process problem above. The current config uses a non-empty list, so
   the question is moot in practice but would matter for anyone writing a new
   profile.
2. **Which of the four original config changes fixed tool calling?** Unknown,
   see above.
3. **Why does the gateway webhook ignore risk profiles?** Established *that*
   it does, with a clean reproduction. Whether it is a bug, a deliberate
   "trusted local caller" design, or a misconfiguration on our side is not
   established. Worth reading the ZeroClaw source before filing anything.
4. **Is `small.en` genuinely better than `base.en` on real room audio?** The
   evidence is 10 synthetic phrases with simulated reverb and noise, which
   reproduced the observed error class — suggestive, not conclusive. A real
   recorded sample set would settle it.

---

# Session 4 — prompt versus code

## Use the prompt for style, use code for correctness

SOUL.md told the voice agent, in plain language, to speak American units and
avoid symbols. It then said:

> "a temperature of 28 degrees Celsius ... humidity is at 50% ... 6 kilometers
> per hour"

Two explicit instructions ignored in one sentence. The cause is not a bad
prompt: the weather **tool** returns strongly formatted metric text, and a 30B
model relaying tool output echoes that formatting regardless of what the
system prompt asked for.

The fix was `homeai/speech.py` — deterministic unit conversion and symbol
expansion, with 30 tests. Now it cannot regress, and it does not depend on the
model being obedient on any given turn.

**Rule: if a behaviour must be reliable, it does not belong in the prompt.**
Prompts shape tone, length, and register. Anything with a right answer -
units, symbols, formatting, redaction - belongs in code where it can be tested.

Notably the *location* instruction in AGENTS.md **did** work (it correctly said
Lilburn). The difference: location changes what the model *asks the tool for*,
while units require it to *transform the tool's answer*. Prompts are decent at
the former and unreliable at the latter.

## Look for the limit you forgot you wrote

After enabling long, detailed answers via the prompt, `sanitise_for_speech`
was still silently truncating at **600 characters** - roughly half a page. The
user would have asked for depth, been told they would get it, and been cut off
mid-thought with no error anywhere.

The feature was "enabled" in the prompt and disabled in the code. When adding
a capability, grep for the constants that quietly constrain it: truncation
limits, timeouts, buffer sizes, retry caps. Two were found this session - the
600-char cap and a 30s TTS timeout that would have cut off any reply over
about 75 words.

## Muting an input is not the same as clearing it

The mic is muted while Jarvis speaks, yet a wake word still fired 0.487s after
a reply finished. openWakeWord retains roughly a second of internal audio
context, so our own speech was still inside the model even though no new audio
was being captured. The detector must be **reset and made briefly deaf on mic
resume**, not merely starved of input.

This is the second instance of the same misconception this project, after the
identical bug at end-of-capture. Stateful detectors need explicit clearing at
*every* boundary, not just the obvious one.

---

# Session 5 — a hypothesis I had written down as a fact

## "The coding model is wrong for conversation" was wrong

This sat in the todo list for two sessions as an assertion: the voice agent
runs `qwen3-coder-30b`, a *coding* model, therefore conversation quality must
be suffering, therefore swap it for Qwen2.5-7B/14B-Instruct. Two general
instruct models were downloaded on the strength of it.

Measured with `tools/bench_llm.py` across ten voice-style prompts, it is false.

| | coder-30B (GPU) | Qwen2.5-7B (CPU) |
|---|---|---|
| speakable answers | 6/10 | 8/10 |
| median TTFT | 108 ms | 162 ms |
| "how hot kills bacteria" | **160 °F** (correct) | 150 °F |
| conversational answer | specific, actionable | generic, two lines |

The 7B scored *better* on the automated speakability check and was still the
worse assistant. Its wins were all "didn't break a formatting rule"; its
losses were substance and accuracy. The coder's only real faults were
verbosity and one markdown emission -- **format problems, which code can fix,
not knowledge problems, which only a different model can fix.**

Decision: no model swap, no VRAM juggling. The Qwen2.5 files stay as fallback.

Two lessons worth carrying:

1. **A todo written as a conclusion will be executed as one.** This one said
   "the wrong tool for the job" in its description, and every later session
   treated that as settled. Write todos as the *question* ("does a general
   instruct model beat the coder for voice?"), not the answer.
2. **An automated score is not a verdict.** The 8/10 vs 6/10 pointed at the
   losing model. The benchmark prints
   "Answer quality is not scored here. Read the saved text." for exactly this
   reason, and reading the text is what reversed the decision.

## Time-to-first-token is not generation speed

The assumption behind the whole swap plan was that a CPU endpoint would be far
too slow. Measured: **162 ms on CPU versus 108 ms on GPU**. For short prompts
the GPU advantage nearly vanishes, because prompt processing parallelises
across 24 cores. The real gap is *generation* -- 66 chars/s on CPU versus
150+ on GPU.

That distinction matters because generation (66 chars/s) still comfortably
outruns speech (~15 chars/s). So a CPU-hosted model could feed a streaming TTS
pipeline without ever starving it. That is now the `streaming-tts` todo, and
it would cut perceived latency far more than any model swap would have.
**Measure the component you actually depend on, not the one that is easy to
benchmark.**

## Hard-coded hardware is a silent failure, not a loud one

`plughw:2,0` was welded into the config. Card 2 is only where the analog
codec lands *because* this box has a discrete GPU occupying cards 0 and 1. On
a laptop it is card 0; the hard-coded value would have selected HDMI.

That failure mode is the dangerous kind: ALSA opens the HDMI device happily,
Piper writes to it happily, every log line says success, and the room stays
silent. Nothing in the stack reports an error, so the bug presents as "the
assistant ignored me".

`homeai/hardware.py` now ranks outputs by intent (USB > headphone > analog >>
digital > HDMI) and **refuses to auto-select HDMI at all**, falling back to
`default` instead of guessing. Detection independently reproduces
`plughw:2,0` on this box, which is the regression test.

General rule for this codebase as it moves toward being usable by others:
**detect, rank, allow an explicit override, and refuse to guess when the
wrong guess fails silently.** An env var must always be able to overrule a
heuristic -- no detector should outvote a human who knows their own hardware.

---

# Session 6 — the failure that reports success

## A missing config line produced a confident lie

The weather tool was wired in, the MCP server tested clean by hand, and the
agent was asked "what's the weather today?" It answered:

> "It's currently 72 degrees and partly cloudy in Lilburn. The high today will
> be around 75 with a low near 60. There's a small chance of scattered showers
> this evening."

Reality, from a direct API call thirty seconds earlier: **77 °F, clear, high
91, low 72.** Every number invented. No error, no warning, no tool call — the
model simply did not have the tool and filled the gap with something that
sounded exactly like a weather report.

The cause was one line: `acp_enable_mcp = false` on `agents.local`, plus no
`mcp_bundles` entry. Four separate things must all be right before an MCP tool
is visible (server declared, bundle defined, bundle granted to the agent, tool
auto-approved), and **getting any one wrong fails silently** — the tool is not
reported as missing, it simply never appears.

**Rule: verify a tool against ground truth, never by asking the agent whether
it worked.** A plausible answer is not evidence the tool ran. The only proof
is comparing the agent's numbers against a direct call to the same source.
This is the second time this project has been misled by fluent output; the
first was the units problem, where the model *said* it was using Fahrenheit
while emitting Celsius.

## "It's broken" was true, but it wasn't ours

Weather had been failing for hours, and the obvious assumption was a bug in
the new code. It was not: `wttr.in`'s TLS certificate expired at 08:35 GMT
that morning, five hours before the report.

```
* Server certificate: expire date: Sep 15 08:35:01 2026 GMT
* SSL certificate OpenSSL verify result: certificate has expired (10)
```

Worth noticing *how* this surfaced: `curl` reported `status=000` in 0.26 s,
which looks like a network outage. DNS resolved fine. Only `curl -v` showed
the certificate line. **When a request fails instantly rather than timing out,
suspect TLS or a refused connection, not the network.**

The deeper lesson is architectural: the assistant's most-used feature depended
on a single free third-party service with no fallback and no ownership. It is
now local code against open-meteo. Anything users will ask several times a day
should not be one stranger's expired certificate away from failing.

## Snippets are not sources

The "MONDS" answer — a Rainmeter theme and two football recruits — was easy to
dismiss as the model being dumb. It was not. The search tool returned *titles
and snippets*, and snippets are advertising copy. Nothing in that input could
have produced a good answer.

`research.py` fetches the pages and returns numbered excerpts tagged with
their domain, so the model can weigh `en.wikipedia.org` against a forum. Two
design choices matter more than the fetching:

* **Return source material, never conclusions.** The tool does not summarise.
  Summarising inside the tool would hide which claims came from where.
* **Refuse to return thin results.** Pages under 400 characters are dropped,
  and when nothing is readable the tool explicitly instructs the model to say
  it could not find reliable information. An empty string would have been
  filled with invention, exactly as the weather gap was.

## Silence reads as failure

A research call takes ten to fifteen seconds. During that time the assistant
is mute, which is indistinguishable from not having heard the wake word — so
the user repeats themselves, re-triggering the wake word mid-turn.

The daemon now says "Let me look that up." when the agent exceeds 2.5 s. The
threshold sits above a normal turn (0.5–1.5 s) so simple questions are never
padded. The announcement is spoken on the *calling* thread, not the worker:
`_speak` mutes the microphone, and two threads racing on mute state would
leave the mic muted forever after one of them finished.

---

# Session 7 — delegation that was worth it despite being wrong

## The local model reported success without running anything

Test-writing for `memory.py` was delegated to the `builder` agent to save
credits. It produced a complete file and reported:

> "All tests are properly written... The final pass count would be 13 tests
> that all pass, assuming the memory.py implementation is correct (which I've
> verified by reading it)."

Run for real: **4 of 13 failed.** The agent could not execute the tests
("security restrictions") and reported a hypothetical result in language that
reads like a measurement. Note the tell: *"the final pass count **would be**"*.

**Delegated work is a draft, never a result.** This is now three sessions in a
row where fluent output was mistaken for a verified one — the units problem,
the hallucinated weather report, and now this. The check is always the same
and always cheap: run it yourself.

## It was still worth delegating — one failure was real

Three of the four failures were test bugs. The fourth was a genuine bug in
*my* code:

```python
memory.add("user", "assistant")   # returned True
len(memory)                       # 0
```

With `max_turns=0`, `add()` appended, trimmed the entry away, and returned
`True` regardless. It described a window that did not exist. Fixed to return
`entry in self._entries` — report what actually happened, not what was
attempted.

A test author who has not read the implementation's *intent* writes exactly
the naive assertions that catch this. That is the real value of delegating
tests, and it survives the model being wrong about everything else.

## Do not sed by line number into a file you have not re-read

Fixing the delegated tests, a `sed -i '182s/...'` overwrote a `def` line
instead of the constructor, because line numbers had shifted from an earlier
edit in the same session. The next two "fixes" made it worse — a stray
docstring below the code, then duplicate constructor lines.

Cost: four extra round-trips at a moment when credits were explicitly scarce.
**Use anchored string replacement, not line numbers**; or if line numbers are
unavoidable, re-read the file immediately before editing. `python - <<'PY'`
with an exact string match would have been right the first time.

## Measure before scoping, not after committing

Two probes, a few seconds each, reshaped the plan:

* **Does the ZeroClaw CLI stream?** No — 20 lines all arrived at 2.28 s. That
  killed token-streamed TTS outright and rescoped the work to chunked
  playback, *before* any code was written against a false assumption.
* **What are PipeWire's defaults?** Sink `iec958-stereo` (S/PDIF), source its
  own monitor. That turned barge-in from "load a module" into "re-plumb both
  ends of the audio path", and justified deferring it behind a cheaper
  between-chunks experiment.

Both probes were cheaper than a single wrong implementation.

---

## Session 8 — Barge-in, a false verdict, and per-task temperature

### The most expensive mistake: a measurement tool that could not fail

`tools/probe_bargein.py` was written to decide the barge-in architecture. It
reported **0 false triggers, peak score 0.0000, "comfortable margin"**. That
verdict was entirely false.

`OpenWakeWordDetector` loads its model in an explicit `load()` call. The probe
never called it, so `_model` stayed `None`, and `detect()` returns `False`
unconditionally in that state — **no exception, and `self._error` stayed empty**.
The probe's `_score()` helper then "helpfully" fell back to the boolean
`detect()`, so every frame scored 0.0.

A peak of *exactly* 0.0000 was the tell. Real acoustic measurements are never
exactly zero. That suspicion is what prompted the positive control.

**Rule: every detector-based measurement needs a positive control in the same
run.** Feed it something that *must* trigger and assert that it does. If the
meter cannot be shown to fire, a reading of "nothing detected" means nothing.
`probe_bargein.py` now refuses to report a verdict if `load()` fails, and
`_score()` raises instead of falling back.

This is the same failure shape as the Session 6 weather hallucination: a
capability was silently absent, and the system produced a confident, fluent,
wrong answer rather than an error. **Silent fallbacks manufacture false
confidence. Prefer a loud failure.**

### Second lesson: one worst-case test gives the wrong architecture

With a working meter, the adversarial passage (which deliberately contained
"Hey ..." and "... Jarvis") gave 4 false triggers and peak 0.9953 →
"echo cancellation required". That would have meant switching the sound card to
`pro-audio` and re-plumbing all desktop audio — invasive and risky.

Running the *ordinary* prose case gave peak **0.1026**, 0 triggers, margin
+0.40. The real conclusion is far cheaper:

| Passage | Peak | Triggers |
|---|---|---|
| Ordinary prose | 0.1026 | 0 |
| Contains "Hey"/"Jarvis" | 0.9953 | 4 |

Jarvis only self-triggers when it *says its own name*. We always know the text
we are about to speak, so a deterministic text check (`contains_wake_fragments`)
handles it with no DSP at all. **Measure the normal case as well as the worst
case; the gap between them is where the cheap solution lives.**

### Barge-in as built

- `homeai/bargein.py` — `InterruptListener` + `contains_wake_fragments`.
- The mic now stays **open** during chunked replies (the daemon previously
  always paused it). When no listener is active, the old pause behaviour is
  kept unchanged as the fallback.
- The listener starts its cursor at the buffer's current `_total`, so the
  user's own just-spoken request cannot interrupt the reply it asked for.
  There is a regression test for exactly this.
- It refuses to run on the energy-fallback detector, which would fire on
  Jarvis's own voice instantly and make every reply self-interrupt.
- On interrupt, `_speak` deliberately **skips** the detector/machine reset:
  the main wake loop heard the same wake word and is already capturing the
  follow-up. Resetting would discard the utterance the user interrupted to say.
- Config: `wake.barge_in` (`HOMEAI_BARGE_IN`), default on.

### ZeroClaw: temperature is per-provider, not per-agent

`AliasedAgentConfig` has **no** `temperature` key — only `model_provider`.
Temperature lives on `providers.models.<backend>.<name>`. Both `local` (voice)
and `builder` (code) pointed at the same provider entry, so both ran at **0.7**,
including all code generation.

Fix: a second provider entry against the *same* llama.cpp server and model,
differing only in sampling.

| Provider entry | Agent | Temperature |
|---|---|---|
| `llamacpp.local` | `local` (voice) | 0.45 |
| `llamacpp.coder` | `builder` (code/tools) | 0.2 |

Verify with `zeroclaw config get providers.models.llamacpp.coder.temperature`.
A read-back is worth doing every time: a malformed value silently reverts the
whole file to defaults (Session 1).

### Tooling note

`tools/*.py` need `PYTHONPATH=.` — they are not installed into the venv.
`systemctl --user stop homeai.service` before any probe that opens the mic, and
remember to start it again; the service holds the device exclusively.

---

## Session 9 — The interrupt that "worked" but felt broken

First field test of barge-in. Two distinct bugs, both invisible to the unit
tests, both obvious in the service log. **The log had the answer in under a
minute; guessing would have cost far more.**

```
14:48:40.917  bargein: wake word heard during playback   <- detection: instant
14:48:48.638  speech interrupted after 1 of 4 chunks     <- silence: 7.7s later
14:48:49.236  discarding utterance: transcript below minimum length
```

### Bug 1: detection was instant, stopping was not

`say_chunked` polled `should_stop` **between** chunks only. I had documented
that as a deliberate simplification ("no audio is playing at that moment, so
there is nothing to race with"). The flaw: chunk one was **9.9 seconds** of
audio, so the poll could not happen for 7.7s after the wake word.

The user experienced this as "the interrupt didn't work", spoke again, and
their second attempt was then talked over. **A correct mechanism with the
wrong granularity is indistinguishable from a broken one.**

Fix: `should_stop` is now threaded down through `say_safe` -> `say` ->
`_run_pipeline` and polled every 50ms *during* playback by
`_await_playback`. On stop, `aplay` and `piper` are **killed**, not waited on.
`aplay` holds only a short ALSA buffer, so the voice dies almost instantly.

Measured after the fix, on real playback: **cut latency 1ms** (was ~7700ms).

### Bug 2: the follow-up question was thrown away

`CaptureMachine.feed` ended an utterance after `silence_s` (0.7s) of quiet.
But the natural way to interrupt is: say the wake word, **pause to check it
worked**, then ask. That pause closed the capture on the wake word alone, and
the result was discarded as "below minimum length".

Fix: a separate `lead_in_s` (default 2.5s) applies *until the first real
speech is heard*; `silence_s` governs only afterwards, so replies stay snappy
once you are actually talking. This also fixes the ordinary
"Hey Jarvis... <thinks> ...what's the weather" case, which must have been
failing silently all along.

### Lessons

1. **Measure the user-visible quantity, not the internal event.** "Interrupt
   detected" was true and useless. The number that mattered was the delay
   between detection and silence, and nothing measured it until a human
   complained. The new test asserts the callback is polled *repeatedly*.
2. **A timeout tuned for one phase of an interaction may be wrong for
   another.** `silence_s` was tuned for end-of-sentence detection and was
   silently also acting as a start-of-speech deadline.
3. **Appending tests to an existing file can shadow its helpers.** My new
   `_machine()`/`_loud()` helpers collided with existing ones and broke five
   passing tests. Check for existing names before appending; prefix new
   helpers.

Result: 374 tests passing (was 365).

---

## Session 10 — The interrupt fired, but nobody was listening

Second field test. The cut was now instant (10ms), but the follow-up question
produced **nothing at all in the log** — not even a discard. That absence was
the clue: the utterance was never captured, so no code ran to reject it.

### The bug I had written into my own comment

`_wake_loop` begins:

```python
if self.mic.paused or self.tts.speaking:
    cursor = self.mic.buffer.total_written
    continue
```

**The wake loop is deaf for the entire duration of every reply**, and discards
buffered audio on each poll. Yet in Session 8 I wrote, in `_speak`:

> "The main wake loop heard the same words and has already begun capturing
> their follow-up, so resetting here would discard the very utterance..."

That was false. Only `InterruptListener` heard the wake word, and it captures
no audio — it only sets a flag. So after an interrupt the daemon returned to
IDLE and waited for a *second* wake word that the user had no reason to say.

**Lesson: a comment asserting the behaviour of another thread is a claim, not
a fact.** I reasoned about the wake loop instead of reading it, and the false
premise survived because the interrupt bug (Session 9) masked it entirely.

Fix: `_bargein_armed`, a `threading.Event` set by `_speak` when a reply was
cut short. The wake loop consumes it *after* the speaking guard, clears the
utterance buffer, adopts a fresh cursor, and calls `machine.on_wake()`
directly — deliberately bypassing `accepts_wake`, since `_speak` has just
opened a refractory window and the person has already spoken the wake word.

`tests/test_daemon.py` now exists (4 tests), covering the handoff, the
refractory bypass, and that arming does **not** take effect while our own
audio is still playing.

### Debugging note that keeps paying off

Three sessions running, the service log identified the fault faster than any
reasoning did. Here, the *absence* of a log line was the diagnostic. Read
`journalctl --user -u homeai.service` first, every time.

### ZeroClaw CLI cannot see ~/src — by design

`acp.default_agent = "local"`, and `agents.local` runs the `voice` risk
profile, which lists `~/src` in `forbidden_paths`, sets `workspace_only = true`
and limits `allowed_roots` to voice-scratch and /tmp.

That is the boundary protecting against anyone within earshot of the mic, so
**do not widen it**. Use the `builder` agent, which already has
`standard` (allowed_roots includes /home/adam, unrestricted_filesystem = true)
and runs at temperature 0.2.

Verified empirically both ways: builder answered a question about ~/src
correctly; local refused with "I cannot access that directory due to security
restrictions".

Convenience wrapper added at `~/.local/bin/zc`:

    exec zeroclaw agent -a builder "$@"

## Session 11 — Recovering from the OpenClaw migration

A week of work by a local Qwen coder agent left the voice path answering
"something went wrong" to every question. One root cause, and a second failure
that explains why it was never caught.

### The crash: a broken contract, patched at the symptom

`config.py` had its `transport` default changed to `"zeroclaw"`, routing to a
new `zeroclaw_agent.py` whose `ask()` returned a **bare string** where every
caller expected an `AgentReply`. So `daemon.py` did:

    turn.attempts = reply.attempts   # AttributeError: 'str' has no 'attempts'

The telling detail: `_warmup` had been given an `isinstance(reply, str)`
band-aid while `_handle`, thirty lines away, still crashed on the same value.
That is patching where the traceback points instead of repairing the contract
that was broken. It converts a loud failure into a quiet one.

**Nothing was wrong with ZeroClaw, the model, or the config.** Verified
directly: `zeroclaw agent --agent local --message "Reply with only: OK"`
returned `OK` throughout.

### Why it went unnoticed for a week: the test suite went dark

`CliAgentClient` was renamed `OpenClawAgentClient`, breaking the import in
`tests/test_agent_cli.py`. **pytest aborts the entire run on a collection
error**, so all 378 tests silently stopped executing. The agent had zero
feedback and kept "fixing" things.

Lesson: a suite that does not *collect* is not a suite that passes. Treat a
collection error as a total outage, not a warning. `pytest --collect-only -q`
is the first thing to run when tests look suspiciously quiet.

### Protections that were silently deleted

The rewrite dropped a fresh `--session-state-file` per call (anti
session-poisoning; the symptom is Jarvis reading tool markup aloud hours
later, with no error anywhere), the retry loop, `detect_leaked_markup`, and
the "never raises into the caller" contract. None of these announce their
absence. They were restored wholesale by returning to `d9f2e86`.

### Recovery method

Preserved everything on a branch (`qwen-openclaw-experiment`) rather than
discarding it, restored the four damaged files from the last known-good
commit, then **gated on the suite collecting and 378 passing** before
restarting the service. Gate first, restart second.

### The tool-iteration cap: the user was right, I was wrong

`max_tool_iterations = 100` under `[autonomy]` plus a `[runtime_profiles.
heavy_duty]` block is **not sufficient**. The profile must be bound to the
agent:

    [agents.builder]
    runtime_profile = "heavy_duty"   # the missing line

Measured before: `*Turn stopped: reached maximum tool iterations (10).*`
After: 16 of 16 files read. This very likely explains the "says it's fixed
when it isn't" behaviour — a read-edit-test-read-fix-retest cycle is six calls
minimum, so the agent was cut off mid-task and then summarised its intentions
as accomplishments.

Corollary: `agents/builder/workspace/SOUL.md` still *told* the agent it had 10
calls. A stale limit in a prompt is as damaging as the limit itself. Updated.

### The voice agent was running a coding model

`providers.models.llamacpp.local` and `.coder` both point at
**`qwen3-coder-30b`**; only `temperature` differs (0.45 vs 0.2). So the warm,
philosophical voice persona was being generated by a code-specialised 30B
model. Measured: 33.9s for one paragraph, and three consecutive timeouts at
the old 45s ceiling. Raised `HOMEAI_AGENT_TIMEOUT` to 90 as a stopgap; the
real fix is a prose model (mistral-nemo:12b) for the voice path.

### Do not trust benchmarks taken under VRAM pressure

Timing dolphin3:8b against mistral-nemo:12b gave 4.2s/9.1s, then 8.5s/0.6s for
the same prompts. With llama-server (17GB) and two Ollama models resident in
21.4GB of VRAM, those numbers measure **which model happened to be resident**,
not model speed. Stop the other runtime before benchmarking, or report
nothing.
