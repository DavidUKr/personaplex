import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


logger = logging.getLogger(__name__)


def _default_kb_path() -> Path:
    return Path(__file__).with_name("knowledge").joinpath("hellotech_kb.json")


def _resolve_kb_path(configured_path: Path) -> Path:
    candidates: list[Path] = [configured_path]
    cwd = Path.cwd()

    # Support running from the repo root even when the installed package in site-packages
    # does not include the bundled knowledge JSON yet.
    candidates.extend([
        cwd / "moshi" / "moshi" / "knowledge" / configured_path.name,
        cwd / "moshi" / "knowledge" / configured_path.name,
        cwd / "knowledge" / configured_path.name,
    ])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return configured_path


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
    kb_json_path: Path = _default_kb_path()
    kb_top_k: int = 6
    kb_relevance_threshold: float = 0.3
    trigger_mode: str = "user"
    trigger_keyword: str = "my supervisor"
    payload_mode: str = "rolling"
    rolling_lines: int = 15
    injection_template: str = ""
    poll_seconds: float = 1.0
    log_dir: Path = Path("./logs/conversations")
    fallback_text: str = (
        "I'm sorry, I don't have that information available right now. "
        "Could you try asking in a different way?"
    )


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
        self._vector_store = None

    def startup(self) -> None:
        if not self.config.enabled:
            return
        if self.config.rolling_lines <= 0:
            raise ValueError("--llm-rolling-lines must be > 0")
        if self.config.poll_seconds <= 0:
            raise ValueError("--llm-poll-seconds must be > 0")
        if self.config.kb_top_k <= 0:
            raise ValueError("--llm-kb-top-k must be > 0")
        if not 0.0 <= self.config.kb_relevance_threshold <= 1.0:
            raise ValueError("--llm-kb-threshold must be between 0 and 1")
        if self.config.trigger_mode == "keyword" and not self.config.trigger_keyword.strip():
            raise ValueError("--llm-trigger-keyword must not be empty when --llm-trigger-mode keyword is used")

        try:
            from langchain_core.documents import Document
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import OpenAIEmbeddings
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM log watcher requires the OpenAI and LangChain dependencies. Install the server dependencies "
                "again after updating requirements."
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

        kb_path = _resolve_kb_path(self.config.kb_json_path)
        if not kb_path.exists():
            raise RuntimeError(f"LLM watcher knowledge base file does not exist: {kb_path}")
        if kb_path != self.config.kb_json_path:
            logger.warning(
                "configured KB path %s was not found; using fallback KB path %s",
                self.config.kb_json_path,
                kb_path,
            )
        kb_entries = json.loads(kb_path.read_text(encoding="utf-8"))
        if not isinstance(kb_entries, list) or not kb_entries:
            raise RuntimeError(f"LLM watcher knowledge base is empty or invalid: {kb_path}")

        docs = []
        for idx, entry in enumerate(kb_entries):
            if not isinstance(entry, dict):
                continue
            title = str(entry.get("title", "")).strip()
            text = str(entry.get("text", "")).strip()
            if not title and not text:
                continue
            docs.append(
                Document(
                    page_content=f"{title}\n\n{text}".strip(),
                    metadata={
                        "url": str(entry.get("url", "")).strip(),
                        "title": title,
                        "chunk_idx": entry.get("chunk_idx"),
                        "kb_idx": idx,
                    },
                )
            )
        if not docs:
            raise RuntimeError(f"LLM watcher knowledge base has no usable entries: {kb_path}")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        self._vector_store = InMemoryVectorStore(embeddings)
        self._vector_store.add_documents(documents=docs)
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

        payload = await self._build_payload(log_path)
        if not payload:
            return
        llm_output = (await self._generate_prompt(payload)).strip()
        if not llm_output:
            fallback = self.config.fallback_text.strip()
            if fallback:
                logger.info("llm watcher empty; injecting fallback text")
                llm_output = fallback
            else:
                return  # specific reason already logged at WARNING in _generate_prompt

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
        if self._client is None or self._vector_store is None:
            raise RuntimeError("OpenAI client is not initialized.")

        query = self._extract_query(payload)
        if not query:
            logger.warning("llm watcher empty: no user query extracted from transcript payload")
            return ""

        retrieved_docs, all_scores = await self._retrieve_docs(query)
        if not retrieved_docs:
            top_score = max(all_scores) if all_scores else 0.0
            logger.warning(
                "llm watcher empty: no KB context above threshold %.2f for query %r "
                "(top score among %d candidates: %.3f)",
                self.config.kb_relevance_threshold,
                query,
                len(all_scores),
                top_score,
            )
            return ""
        context_str = self._build_context_string(retrieved_docs)

        def _request() -> str:
            response = self._client.responses.create(
                model=self.config.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": self._system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": (
                                f"TRANSCRIPT CONTEXT:\n{payload}\n\n"
                                f"RETRIEVED CONTEXT:\n{context_str}\n\n"
                                f"USER QUESTION: {query}\n\n"
                                "Answer the user's question using only the retrieved context above. "
                                "If the retrieved context does not answer the question, return an empty response."
                            ),
                        }],
                    },
                ],
            )
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str):
                return output_text
            return str(output_text or "").strip()

        result = await asyncio.to_thread(_request)
        if not result.strip():
            kept_top = max(
                (s for s in all_scores if s >= self.config.kb_relevance_threshold),
                default=0.0,
            )
            logger.warning(
                "llm watcher empty: KB returned %d docs (top score %.3f) but model %r produced no answer",
                len(retrieved_docs),
                kept_top,
                self.config.model,
            )
        return result

    async def _retrieve_docs(self, query: str) -> tuple[list, list[float]]:
        def _search() -> tuple[list, list[float]]:
            assert self._vector_store is not None
            retrieved_with_scores = self._vector_store.similarity_search_with_score(
                query,
                k=self.config.kb_top_k,
            )
            all_scores = [float(score) for _, score in retrieved_with_scores]
            kept = [
                doc for doc, score in retrieved_with_scores
                if score >= self.config.kb_relevance_threshold
            ]
            return kept, all_scores

        return await asyncio.to_thread(_search)

    def _extract_query(self, payload: str) -> str:
        user_lines: list[str] = []
        ignored_lines: list[str] = []

        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if line.startswith("[user ignored]"):
                ignored_lines.append(line[len("[user ignored]"):].strip())
            elif line.startswith("[user]"):
                user_lines.append(line[len("[user]"):].strip())

        if self.config.trigger_mode == "keyword":
            parts = []
            if user_lines:
                parts.append(user_lines[-1])
            parts.extend(text for text in ignored_lines if text)
            return "\n".join(part for part in parts if part).strip()

        if user_lines:
            return user_lines[-1]
        if ignored_lines:
            return ignored_lines[-1]
        return ""

    def _build_context_string(self, docs: list) -> str:
        context_parts = []
        for idx, doc in enumerate(docs, start=1):
            context_parts.append(
                f"[Source {idx}] {doc.metadata.get('title', '')} - {doc.metadata.get('url', '')}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def _format_injection_text(self, llm_output: str) -> str:
        template = self.config.injection_template.strip()
        if not template:
            return llm_output
        return template.replace("{prompt}", llm_output)
