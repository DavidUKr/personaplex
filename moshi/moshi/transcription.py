import asyncio
import concurrent.futures
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _sanitize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_words(text: str) -> list[str]:
    return [word for word in re.split(r"\s+", text.strip()) if word]


def _drop_overlapping_prefix(previous_text: str, current_text: str, max_words: int = 32) -> str:
    previous_words = _split_words(previous_text.lower())
    current_words = _split_words(current_text)
    current_lower = [word.lower() for word in current_words]
    max_overlap = min(len(previous_words), len(current_words), max_words)

    overlap = 0
    for word_count in range(max_overlap, 0, -1):
        if previous_words[-word_count:] == current_lower[:word_count]:
            overlap = word_count
            break

    if overlap == 0:
        return current_text
    return " ".join(current_words[overlap:])


class ConversationLogger:
    def __init__(self, log_dir: Path, session_id: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = log_dir / f"{timestamp}_{session_id}.log"
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write_entry(self, tag: str, text: str) -> None:
        cleaned = _sanitize_text(text)
        if not cleaned:
            return
        async with self._lock:
            self._fh.write(f"[{tag}] {cleaned}\n")
            self._fh.flush()

    async def close(self) -> None:
        async with self._lock:
            self._fh.flush()
            self._fh.close()


class ModelTextLogger:
    def __init__(self, conversation_logger: ConversationLogger):
        self.conversation_logger = conversation_logger
        self._buffer = ""

    async def append_piece(self, piece: str) -> None:
        self._buffer += piece
        flush_index = self._find_flush_index()
        if flush_index is None:
            return
        segment = self._buffer[:flush_index]
        self._buffer = self._buffer[flush_index:]
        await self.conversation_logger.write_entry("model", segment)

    async def flush(self) -> None:
        if not self._buffer.strip():
            self._buffer = ""
            return
        await self.conversation_logger.write_entry("model", self._buffer)
        self._buffer = ""

    def _find_flush_index(self) -> Optional[int]:
        for marker in ".?!\n":
            index = self._buffer.rfind(marker)
            if index != -1:
                return index + 1
        if len(self._buffer) > 160:
            last_space = self._buffer.rfind(" ")
            if last_space > 0:
                return last_space
        return None


@dataclass
class TranscriptionConfig:
    enabled: bool = False
    model_id: str = "distil-whisper/distil-large-v3"
    chunk_seconds: float = 6.0
    overlap_seconds: float = 1.5
    log_dir: Path = Path("./logs/conversations")


class TranscriptionService:
    def __init__(self, config: TranscriptionConfig, device: torch.device):
        self.config = config
        self.device = device
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._pipeline = self._load_pipeline()

    def create_session(self, conversation_logger: ConversationLogger, sample_rate: int) -> "SessionTranscriber":
        return SessionTranscriber(
            service=self,
            conversation_logger=conversation_logger,
            sample_rate=sample_rate,
            chunk_seconds=self.config.chunk_seconds,
            overlap_seconds=self.config.overlap_seconds,
        )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _load_pipeline(self):
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "User transcription requires the 'transformers' package. "
                "Install the server dependencies again after updating requirements."
            ) from exc

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        device_arg: int | str = self.device.index if self.device.type == "cuda" and self.device.index is not None else (0 if self.device.type == "cuda" else -1)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_arg,
        )

    def transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        result = self._pipeline(
            {"array": audio, "sampling_rate": sample_rate},
            batch_size=1,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe"},
        )
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = ""
        return _sanitize_text(text)


class SessionTranscriber:
    def __init__(
        self,
        service: TranscriptionService,
        conversation_logger: ConversationLogger,
        sample_rate: int,
        chunk_seconds: float,
        overlap_seconds: float,
    ):
        self.service = service
        self.conversation_logger = conversation_logger
        self.sample_rate = sample_rate
        self.chunk_samples = max(1, int(chunk_seconds * sample_rate))
        self.overlap_samples = max(0, int(overlap_seconds * sample_rate))
        self.queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=32)
        self.closed = False
        self.task: Optional[asyncio.Task[None]] = None
        self._pending_audio = np.zeros((0,), dtype=np.float32)
        self._previous_window_text = ""

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    def submit_pcm(self, pcm: np.ndarray) -> None:
        if self.closed:
            return
        flattened = np.asarray(pcm, dtype=np.float32).reshape(-1).copy()
        if flattened.size == 0:
            return
        try:
            self.queue.put_nowait(flattened)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(flattened)
            except asyncio.QueueFull:
                logger.warning("Dropping transcription audio chunk because queue is still full.")

    async def close(self) -> None:
        self.closed = True
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(None)
        if self.task is not None:
            await self.task

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                break
            if self._pending_audio.size == 0:
                self._pending_audio = item
            else:
                self._pending_audio = np.concatenate((self._pending_audio, item))
            while self._pending_audio.size >= self.chunk_samples + self.overlap_samples:
                window = self._pending_audio[: self.chunk_samples + self.overlap_samples]
                await self._transcribe_window(window)
                self._pending_audio = self._pending_audio[self.chunk_samples :]

        if self._pending_audio.size > 0:
            await self._transcribe_window(self._pending_audio)
            self._pending_audio = np.zeros((0,), dtype=np.float32)

    async def _transcribe_window(self, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            self.service.executor,
            self.service.transcribe_sync,
            audio,
            self.sample_rate,
        )
        if not text:
            return
        deduped = _sanitize_text(_drop_overlapping_prefix(self._previous_window_text, text))
        self._previous_window_text = text
        if deduped:
            await self.conversation_logger.write_entry("user", deduped)
