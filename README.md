# Home AI Voice Assistant

A local voice assistant that listens for a wake word, transcribes speech, sends it to the ZeroClaw agent for processing, and speaks the response. It runs entirely on the local machine without internet connectivity for the core functionality.

## Architecture

Each module is independently testable and has no hardware dependency in its
unit tests.

- `config.py` - Central configuration; every setting is environment-overridable
- `daemon.py` - Orchestrator wiring audio, STT, TTS, memory, and the agent
- `hardware.py` - ALSA device discovery and ranking, so no device is hard-coded
- `audio.py` - Capture and playback via PortAudio, with a cursor-based ring buffer
- `wake.py` - Wake word detection (openWakeWord), with an energy-based fallback
- `bargein.py` - Interrupting a reply by speaking the wake word during playback
- `stt.py` - Speech-to-text using whisper.cpp
- `tts.py` - Text-to-speech using Piper, including chunked interruptible speech
- `speech.py` - Makes text speakable: units, symbols, markdown, sentence splitting
- `safety.py` - Validates utterances and sanitises responses before they are spoken
- `memory.py` - Bounded, self-expiring conversation context between turns
- `agent_cli.py` - Default subprocess transport to ZeroClaw
- `agent_client.py` - Alternative HTTP transport via the ZeroClaw gateway
- `mcp_server.py` - Exposes local tools to the agent over JSON-RPC stdio
- `weather.py` - Current conditions and forecast via open-meteo
- `research.py` - Search, fetch, and extract sourced excerpts from the web
- `transcript.py` - Append-only JSONL turn log (written outside the repo)

## External dependencies

Two native components are built outside this repository and are not committed
(see `.gitignore`):

- **whisper.cpp** - speech recognition, with a `small.en` model
- **Piper** - speech synthesis, with a voice model

Both are expected under `vendor/`. Paths are configurable; see `config.py`.

## Setup

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

Start the voice assistant:
```bash
.venv/bin/python -m homeai.daemon
```

## Testing

Run the test suite:
```bash
.venv/bin/python -m pytest tests/
```

## Agent transport

There are two transports: `cli` (default) and `http`. The `cli` transport is the default because it enforces the agent's risk profile and is roughly ten times faster. The transport is selected with the `HOMEAI_AGENT_TRANSPORT` environment variable.

## Configuration

Every setting in the configuration can be overridden by an environment variable prefixed with `HOMEAI_`. For example, `HOMEAI_SAMPLE_RATE` overrides the audio sample rate.

Secrets are read from a local `.env` file, which is not committed. The only
secret currently used is `HOMEAI_AGENT_TOKEN`, required by the `http`
transport; the default `cli` transport does not need it.

## Licence

This project is source-available under the **Business Source License 1.1**, not
an open source licence. See [LICENSE](LICENSE) for the authoritative terms.

In short:

- Personal, family, and household use is permitted.
- Internal non-commercial use within an organisation is permitted.
- Offering it to third parties as a hosted or managed service, or otherwise
  providing it commercially, requires a commercial licence from the Licensor.
- Non-production use - reading, modifying, and experimenting with the code - is
  permitted for anyone.
- On **2030-09-15**, each released version converts automatically to the
  **Apache License, Version 2.0**.

Copyright (c) 2026 Adam Wosotowsky.