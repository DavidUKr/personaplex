import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


logger = logging.getLogger(__name__)


def _should_include_log_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return not stripped.startswith("[prompt]")


@dataclass
class LLMWatcherConfig:
    enabled: bool = False
    model: str = "gpt-5-nano"
    system_prompt_file: Path = Path(__file__).with_name("llm_sys_prompt.txt")
    trigger_mode: str = "user"
    trigger_keyword: str = "let me check"
    payload_mode: str = "rolling"
    rolling_lines: int = 15
    injection_template: str = ""
    poll_seconds: float = 1.0
    log_dir: Path = Path("./logs/conversations")
    fallback_text: str = (
        "I'm sorry, I don't have that information available right now. "
        "Could you try asking in a different way?"
    )
    keyword_debounce_seconds: float = 2.0


@dataclass
class _LogReadState:
    offset: int = 0
    partial_line: str = ""


class OpenAILogPromptWatcher:
    def __init__(
        self,
        config: LLMWatcherConfig,
        get_target_log_path: Callable[[], Optional[Path]],
        queue_live_prompt: Callable[[str, str], bool],
        start_keyword_wait: Callable[[Path], bool],
        cancel_keyword_wait: Callable[[Path], None],
        can_start_keyword_wait: Callable[[Path], bool],
    ):
        self.config = config
        self.get_target_log_path = get_target_log_path
        self.queue_live_prompt = queue_live_prompt
        self.start_keyword_wait = start_keyword_wait
        self.cancel_keyword_wait = cancel_keyword_wait
        self.can_start_keyword_wait = can_start_keyword_wait
        self._states: dict[Path, _LogReadState] = {}
        self._client = None
        self._system_prompt = ""

    def startup(self) -> None:
        if not self.config.enabled:
            return
        if self.config.rolling_lines <= 0:
            raise ValueError("--llm-rolling-lines must be > 0")
        if self.config.poll_seconds <= 0:
            raise ValueError("--llm-poll-seconds must be > 0")
        if self.config.trigger_mode == "keyword" and not self.config.trigger_keyword.strip():
            raise ValueError("--llm-trigger-keyword must not be empty when --llm-trigger-mode keyword is used")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM log watcher requires the 'openai' package. Install the server dependencies again after "
                "updating requirements."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("LLM log watcher requires OPENAI_API_KEY to be set.")

        prompt_path = self.config.system_prompt_file
        if not prompt_path.exists():
            raise RuntimeError(f"LLM system prompt file does not exist: {prompt_path}")
        self._system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not self._system_prompt:
            raise RuntimeError(f"LLM system prompt file is empty: {prompt_path}")

        self._client = OpenAI(api_key=api_key)

    async def run(self) -> None:
        if not self.config.enabled:
            return
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("llm log watcher poll failed: %s", exc)
            await asyncio.sleep(self.config.poll_seconds)

    async def _poll_once(self) -> None:
        log_path = self.get_target_log_path()
        if log_path is None or not log_path.exists():
            return
        state = self._states.setdefault(log_path, _LogReadState())
        stat = log_path.stat()
        if stat.st_size < state.offset:
            state.offset = 0
            state.partial_line = ""

        with log_path.open("r", encoding="utf-8") as fh:
            fh.seek(state.offset)
            appended_text = fh.read()
            state.offset = fh.tell()

        if not appended_text:
            return

        buffer = state.partial_line + appended_text
        if buffer.endswith("\n"):
            complete_text = buffer
            state.partial_line = ""
        else:
            split_at = buffer.rfind("\n")
            if split_at == -1:
                state.partial_line = buffer
                return
            complete_text = buffer[: split_at + 1]
            state.partial_line = buffer[split_at + 1 :]

        new_lines = [line.strip() for line in complete_text.splitlines() if line.strip()]
        if not new_lines:
            return
        if self.config.trigger_mode == "user" and not any(line.startswith("[user]") for line in new_lines):
            return
        if self.config.trigger_mode == "keyword":
            keyword_triggered = any(
                line.startswith("[model]") and self.config.trigger_keyword.strip().lower() in line.lower()
                for line in new_lines
            )
            if not keyword_triggered:
                return
            if not self.can_start_keyword_wait(log_path):
                return

        logger.warning(
            "llm watcher triggered (mode=%s, log=%s)",
            self.config.trigger_mode,
            log_path.name,
        )

        # Keyword detection can fire before the customer's in-flight utterance
        # has been committed to the conversation log by the transcription
        # pipeline. Wait a beat so _build_payload reads the question line, not
        # only the greeting that preceded it.
        if (
            self.config.trigger_mode == "keyword"
            and self.config.keyword_debounce_seconds > 0
        ):
            await asyncio.sleep(self.config.keyword_debounce_seconds)

        payload = await self._build_payload(log_path)
        if not payload:
            return
        llm_output = (await self._generate_prompt(payload)).strip()
        if not llm_output:
            fallback = self.config.fallback_text.strip()
            if fallback:
                logger.info("llm log watcher returned empty output; using fallback text")
                llm_output = fallback
            else:
                logger.warning("llm log watcher returned empty output; skipping injection")
                return

        logger.info("llm log watcher output: %s", llm_output)
        injection_text = self._format_injection_text(llm_output)
        prompt_source = "llm_keyword" if self.config.trigger_mode == "keyword" else "llm_watcher"
        if self.config.trigger_mode == "keyword":
            if not self.start_keyword_wait(log_path):
                logger.warning(
                    "generated llm prompt but session no longer eligible for keyword wait; dropping injection"
                )
                return
        if self.queue_live_prompt(injection_text, prompt_source):
            logger.info("queued llm-generated live prompt")
        else:
            logger.warning("generated llm prompt but no active session is running; dropping injection")
            if self.config.trigger_mode == "keyword":
                self.cancel_keyword_wait(log_path)

    async def _build_payload(self, log_path: Path) -> str:
        if self.config.payload_mode == "full":
            def _read_full_text() -> str:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                kept = [line for line in lines if _should_include_log_line(line)]
                return "\n".join(kept)

            return await asyncio.to_thread(_read_full_text)

        def _read_rolling_lines() -> str:
            lines = log_path.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if _should_include_log_line(line)]
            return "\n".join(kept[-self.config.rolling_lines :])

        return await asyncio.to_thread(_read_rolling_lines)

    async def _generate_prompt(self, payload: str) -> str:
        if self._client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        def _request() -> str:
            system_prompt = self._system_prompt
            if self.config.trigger_mode == "keyword":
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f'The handoff keyword is "{self.config.trigger_keyword.strip()}". '
                    "When that phrase appears in a [model] line, answer the user's unresolved question "
                    "using the transcript context, including any [user ignored] follow-up lines."
                )
            response = self._client.responses.create(
                model=self.config.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": payload}],
                    },
                ],
            )
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str):
                return output_text
            return str(output_text or "").strip()

        return await asyncio.to_thread(_request)

    def _format_injection_text(self, llm_output: str) -> str:
        template = self.config.injection_template.strip()
        if not template:
            return llm_output
        return template.replace("{prompt}", llm_output)
