# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo is split into two top-level deliverables plus shared assets:

- `moshi/` — Python package (`moshi-personaplex`, distributed as `moshi.*` modules). The runtime entry points are `python -m moshi.server` (live WebSocket server) and `python -m moshi.offline` (offline wav-in/wav-out). Model code lives in `moshi/moshi/models/` (Mimi audio codec in `compression.py`, the LM and streaming generator in `lm.py`, weight loaders in `loaders.py`), reusable layers in `moshi/moshi/modules/`, and CLI/server/transcription/watcher orchestration in `moshi/moshi/server.py`, `transcription.py`, and `llm_prompt_watcher.py`.
- `client/` — Vite + React + Tailwind frontend (TypeScript). `client/src/app.tsx` mounts a single `Queue` route; `client/src/pages/Conversation/` is the actual live UI and contains the WebSocket protocol glue, audio worklet hookups (`audio-processor.ts`, `decoder/`), and the model-param controls. The wire protocol is defined in `client/src/protocol/`.
- `assets/` — demo media and example prompts (`assets/test/prompt_service.txt`, sample wavs used by the offline script).

The README at the repo root is the primary user-facing doc and the source of truth for CLI flags and prompting guidance. Mirror its language when writing user-visible help text.

## Backend Architecture (read before changing `server.py`)

The server is a single long-lived `aiohttp` process that holds one `ServerState` (defined in `server.py`). `ServerState` owns the `MimiModel` audio codec, the LM (`LMGen`), and any optional **transcription service** and **LLM watcher**. There is one model loaded into GPU memory; the `asyncio.Lock` on `ServerState` serializes live sessions — assume only one chat WebSocket runs at a time. Streaming state (`mimi.streaming_forever`, `lm_gen.streaming_forever`) is established once at startup and reused per session.

Three loosely coupled subsystems plug into `ServerState` and you should preserve those seams:

1. **Live chat (`handle_chat`)** — bidirectional WebSocket. Drains an `asyncio.Queue[PromptCommand]` between turns; live prompts queued via `queue_live_prompt(...)` are injected at safe boundaries (`LIVE_PROMPT_BOUNDARY_STREAK` / `LIVE_PROMPT_MAX_STEPS` constants govern when injection is allowed). User audio is paused while a prompt is pending — that is intentional and tests/keyword mode rely on it.
2. **Transcription (`transcription.py`)** — when `--enable-transcription` is on, a `SessionTranscriber` runs background ASR on user audio in overlapping windows and a `ConversationLogger` writes per-session UTF-8 logs with tagged lines (`[initial_prompt]`, `[user]`, `[user ignored]`, `[model]`, `[prompt]`). Other subsystems read the log file rather than calling into the transcriber directly.
3. **LLM log watcher (`llm_prompt_watcher.py`)** — polls the active session log, sends a rolling or full payload to OpenAI, and feeds the response back through `queue_live_prompt_with_source`. It has two trigger modes:
   - `user` — fires on any newly appended `[user]` line.
   - `keyword` — fires only when a `[model]` line contains the supervisor phrase, then **pauses PersonaPlex** (`start_keyword_wait` / `cancel_keyword_wait`) until the supervisor answer is injected. While paused, transcribed user speech is tagged `[user ignored]` instead of `[user]`.

When you add functionality that ties these together, keep the callback indirection in `OpenAILogPromptWatcher.__init__` — the watcher must not import server state directly.

### Sampling profiles & session overrides

`PROFILE_PRESETS` in `server.py` defines four `SamplingConfig` values (`default`, `predictable`, `convservative` [note the misspelling — preserve it; it's part of the public CLI surface], `deterministic`). CLI flags `--temp-text`, `--topk-text`, `--temp-audio`, `--topk-audio`, `--seed`, `--greedy/--no-greedy` override the profile at startup. The frontend's `ModelParams` UI passes the same values per-session as WebSocket query params; `resolve_session_sampling_config` honours them **only when `--session-params-override` is enabled** (the default — `--no-session-params-override` makes startup values authoritative).

### Preset bundles

`_apply_bundle_defaults` in `server.py` lets `--default-keyword` and `--default_watcher` expand into multi-flag presets, but only for flags the user did not pass explicitly (`_cli_flag_present` does shallow argv inspection). If you add a preset, follow the same pattern: define the bundle, list its fields in `bundle_fields`, and let real CLI flags win.

## Common Commands

### Backend
```bash
pip install moshi/.                                    # install editable-ish from repo root
python -m moshi.server                                 # minimal local launch (default profile)
python -m moshi.server --default-keyword               # full keyword-supervisor bundle (LLM watcher, gpt-5, static dist)
python -m moshi.server --default_watcher               # watcher + transcription, no keyword pause
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"   # required for browser mic access
python -m moshi.offline --help                         # offline wav→wav inference; see README for full args
```
There is no committed Python test suite. If you add tests, place them under `moshi/tests/`. Linting uses `flake8` config in `moshi/setup.cfg` (max line length 120, ignores `E203,E704`).

### Frontend (all run from `client/`)
```bash
npm install
npm run dev       # vite dev server (requires ./cert.pem and ./key.pem — vite.config.ts enforces https)
npm run build     # tsc + vite build; run this before opening a PR
npm run lint      # eslint
npm run prettier  # format
```
Vite proxies `/api/*` to `VITE_QUEUE_API_URL` when set. `VITE_QUEUE_API_PATH` is required at runtime (`src/env.ts` throws if missing).

### Docker / compose
The root `Dockerfile` builds a CUDA 12.4 image with `uv`; `docker-compose.yaml` exposes 8998 and reserves one NVIDIA GPU. `NO_TORCH_COMPILE=1` is set in compose to skip `torch.compile` warmup.

## Conventions

- Python: 4-space indent, 120-char line limit, snake_case modules, PascalCase classes. The codebase prefers small dataclasses (`SamplingConfig`, `LLMWatcherConfig`, `TranscriptionConfig`) for cross-subsystem config — extend these rather than passing loose kwargs.
- Frontend: 2-space indent, semicolons, double quotes, trailing commas, Tailwind-aware sorting (see `client/.prettierrc.json`). PascalCase components (`ModelParams.tsx`), camelCase hooks (`useSocket.ts`).
- Commit subjects are short and imperative; Conventional Commit prefixes (`fix:`, `docs:`) appear but aren't required.
- The misspelled `convservative` profile name is intentional — referenced from CLI help, README, and the frontend toggle. Do not "fix" it without a coordinated rename.
