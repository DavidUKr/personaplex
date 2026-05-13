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

### Live-prompt injection — gotchas

`inject_prompt` in `server.py` runs whenever the prompt queue drains at a safe boundary. Three things to know:

1. **Step only the new addendum, never the cumulative system prompt.** `inject_prompt` encodes the freshly-prefixed `injected_prompt_text` (with `wrap_with_system_tags`) and calls `lm_gen.step_text_prompt_tokens(...)` on just that. Re-stepping the original `<system>` block makes the model believe a new session started and re-greet the customer.
2. **Stepping text tokens does not emit audio.** `_step_text_prompt_core` in `models/lm.py` feeds zero audio frames while forcing the agent-text channel to the prompt tokens. The model "thinks" it said the prompt silently. After the audio-silence padding, normal generation resumes. This is why the model sometimes follows an injection with wrap-up phrases (*"Thank you for waiting. Let me know if you need anything else."*) instead of actually relaying the answer — open follow-up item.
3. The `live_prompt_prefix` default is `Relay this answer to the customer in full:`. Build_keyword_mode_prompt's appended instruction explicitly tells Moshi to expect that prefix and read the body aloud word-for-word. If you change the prefix, update the instruction in `build_keyword_mode_prompt` to match.

### Watcher logging visibility

`llm_prompt_watcher.py` uses a plain `logging.getLogger(__name__)` with no handler. The watcher's `INFO` records get dropped silently — only `WARNING` and above reach stderr via Python's `lastResort` handler. When adding diagnostics that need to be visible to the operator on EC2 stderr, use `logger.warning(...)`. Reserve `logger.info(...)` for things you don't mind losing.

Existing WARNING points (use them as the model when adding more): the `_poll_once` "llm watcher triggered (mode=…, log=…)" line that fires on every successful trigger, and the empty-output diagnostics in `_generate_prompt` that distinguish *no user query / no KB context above threshold / KB hit but model returned empty*.

### Keyword-mode flow (door_qa_kb)

When `--llm-trigger-mode keyword` (and `--default-keyword`):

1. `LLMWatcherConfig.trigger_keyword` defaults to `"let me check"` (case-insensitive substring against `[model]` lines). It is **only** the watcher's detection substring; the canonical phrase Moshi is told to say is hardcoded as `Let me check on that.` inside `build_keyword_mode_prompt`. Don't re-couple them — that's why `build_keyword_mode_prompt(base_prompt: str)` no longer takes the keyword as a parameter.
2. `ModelTextLogger.flush_phrases` is also seeded from `trigger_keyword` (see `flush_phrases = (self._llm_trigger_keyword,)` in `server.py`). This makes the conversation log flush the moment Moshi emits the keyword, so the watcher can detect it without waiting for a sentence terminator.
3. On detection, the watcher logs the trigger WARNING, then waits `LLMWatcherConfig.keyword_debounce_seconds` (default `2.0`, `--llm-keyword-debounce`) so the transcription pipeline can commit any in-flight `[user]` line. Then `_build_payload` reads the log fresh.
4. If `_generate_prompt` returns empty (no user query / no KB hit / model said empty), `LLMWatcherConfig.fallback_text` is injected so the customer isn't left hanging. Pass `--llm-fallback-text ""` to restore the legacy skip-injection behavior.
5. `start_keyword_wait` is only called after a non-empty `_generate_prompt` (including the fallback path) — so while the OpenAI call is in flight, `is_keyword_wait_active()` is still false and the model's audio is unpaused. The keyword-wait flag pauses user audio only between injection commit and turn resumption.

### Branch differences: `main` vs `door_qa_kb`

These two branches share the same architecture but the watcher is materially different. Cherry-pick with care.

| Area | `main` | `door_qa_kb` |
|---|---|---|
| `_generate_prompt` | Single OpenAI call. Only empty path: model returns empty. | RAG. Calls `_extract_query` → `_retrieve_docs` (LangChain `InMemoryVectorStore` + `OpenAIEmbeddings` "text-embedding-3-small") → adds `RETRIEVED CONTEXT` to the user message before the OpenAI call. Three empty paths (no query / no KB hit / model empty), each logged at WARNING with score diagnostics. |
| `LLMWatcherConfig` extras | — | `kb_json_path`, `kb_top_k`, `kb_relevance_threshold`. |
| CLI flags | base set | adds `--llm-kb-json`, `--llm-kb-top-k`, `--llm-kb-threshold`. |
| `llm_sys_prompt.txt` | older "produce one concise live prompt" wording | rewritten to "produce one concise answer for the on-call agent to read aloud" + KB-only rules. |
| `_retrieve_docs` / `_extract_query` | not present | door-side helpers. |
| Knowledge base assets | not present | `moshi/moshi/knowledge/hellotech_kb.json` (resolved via `_resolve_kb_path` fallback chain). |
| Static dist path in `--default-keyword` | unset | `--static /home/ubuntu/personaplex/client/dist` (door's deploy layout). |

Everything else — `build_keyword_mode_prompt`, `inject_prompt`, the live-prompt prefix, `fallback_text`, `keyword_debounce_seconds`, the trigger WARNING log, the removal of the dead `--live-prompt-mode` flag — is identical on both branches as of `2026-05-13`. Workflow: fixes land on door_qa_kb first; if they don't touch KB-specific code, cherry-pick to `main`. KB-aware fixes stay door-only.

### Buffer & flush map (useful when debugging stuck/duplicated output)

When a log entry, audio chunk, or prompt looks "stuck", check the layer that owns the relevant buffer:

- `ModelTextLogger._buffer` (`transcription.py`) buffers `[model]` pieces until: (a) a `flush_phrases` substring appears (case-insensitive), (b) the last char is `.`/`?`/`!`/`\n` and `_find_flush_index` returns at the marker, or (c) buffer length exceeds 160 chars (then splits at the last space). `await model_text_logger.flush()` is the only forced-flush path; called before `inject_prompt` and at session shutdown.
- `_LogReadState.partial_line` (`llm_prompt_watcher.py`) buffers the tail of a log read that didn't end on `\n`. If the log file never gets a trailing newline (e.g., process killed mid-write), the tail will sit in this buffer until the next poll completes the line.
- `pending_prompt_commands` (`opus_loop` local) buffers `PromptCommand`s drained from `session_prompt_queue`. They are only flushed during `prompt_interrupt_active` once `boundary_streak >= LIVE_PROMPT_BOUNDARY_STREAK` or `boundary_steps >= LIVE_PROMPT_MAX_STEPS`. If the model never reaches a boundary, prompts can sit here.
- `all_pcm_data` (`opus_loop` local) accumulates user PCM frames until at least `frame_size` (`= mimi.sample_rate / mimi.frame_rate`) samples are available. Reset to `None` whenever a prompt interrupt or keyword wait fires; that means user speech currently being buffered gets dropped on interrupt.
- `SessionTranscriber` runs `--transcription-chunk-seconds` (default `6.0`) windows with `--transcription-overlap-seconds` (default `1.5`) overlap. A `[user]` line is only written when a chunk completes; in-flight audio can lag the actual speech by up to ~chunk-seconds. The keyword debounce (`keyword_debounce_seconds=2.0` default) mitigates this for the watcher, not for anything else that reads the log.
- `opus_writer` / `opus_reader` (sphn) hold the WebSocket-side opus codec buffer. Audio in/out flow through these; if frontend reports silence, check whether `opus_reader.read_pcm()` was drained by an interrupt path (`_ = opus_reader.read_pcm()` shows up in both `drain_prompt_queue` and the `prompt_interrupt_active` branch).
- Streaming KV cache in `lm_gen` itself (`streaming_forever(1)`). Not user-controlled — but recall that `inject_prompt` only steps the *new* addendum on purpose; the KV cache already holds the original system prompt and the conversation history.

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
