# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
from contextlib import suppress
import random
import os
from pathlib import Path
import tarfile
import secrets
import sys
from typing import Literal, Optional
from uuid import uuid4

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .client_utils import make_log, colorize
from .llm_prompt_watcher import LLMWatcherConfig, OpenAILogPromptWatcher
from .models import loaders, MimiModel, LMModel, LMGen
from .transcription import (
    ConversationLogger,
    ModelTextLogger,
    SessionTranscriber,
    TranscriptionConfig,
    TranscriptionService,
)
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]
TEXT_TOKEN_EOS = 2
TEXT_TOKEN_PAD = 3
LIVE_PROMPT_BOUNDARY_STREAK = 2
LIVE_PROMPT_MAX_STEPS = 48


@dataclass
class PromptCommand:
    text: str
    source: str = "manual"

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def apply_live_prompt_prefix(text: str, prefix: str) -> str:
    cleaned = text.strip()
    normalized_prefix = prefix.strip()
    if not cleaned:
        return cleaned
    if not normalized_prefix:
        return cleaned
    if cleaned.startswith(normalized_prefix):
        return cleaned
    return f"{normalized_prefix} {cleaned}"


def build_keyword_mode_prompt(base_prompt: str, trigger_keyword: str) -> str:
    keyword = trigger_keyword.strip()
    instruction = (
        "If the user asks a question, "
        f'say exactly "Let me check with {keyword}." '
        f"Then wait for new system information before answering. You will receive [SYSTEM PROMPT]."
        "Do not guess or invent facts while waiting."
    )
    cleaned_base_prompt = base_prompt.strip()
    if not cleaned_base_prompt:
        return instruction
    if instruction in cleaned_base_prompt:
        return cleaned_base_prompt
    return f"{cleaned_base_prompt}\n{instruction}"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock
    transcription_service: TranscriptionService | None
    transcription_config: TranscriptionConfig
    llm_watcher: OpenAILogPromptWatcher | None

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False, live_prompt_mode: str = "append",
                 live_prompt_prefix: str = "[SYSTEM PROMPT]:",
                 transcription_config: TranscriptionConfig | None = None,
                 llm_watcher_config: LLMWatcherConfig | None = None):
        llm_watcher_config = llm_watcher_config or LLMWatcherConfig()
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.live_prompt_mode = live_prompt_mode
        self.live_prompt_prefix = live_prompt_prefix
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.active_prompt_queue: Optional[asyncio.Queue[PromptCommand]] = None
        self.active_conversation_log_path: Optional[Path] = None
        self.last_conversation_log_path: Optional[Path] = None
        self._keyword_wait_active = False
        self._keyword_wait_log_path: Optional[Path] = None
        self._llm_trigger_mode = llm_watcher_config.trigger_mode
        self._llm_trigger_keyword = llm_watcher_config.trigger_keyword
        self.transcription_service = None
        if self.transcription_config.enabled:
            self.transcription_service = TranscriptionService(self.transcription_config, self.device)
        self.llm_watcher = None
        if llm_watcher_config.enabled:
            self.llm_watcher = OpenAILogPromptWatcher(
                llm_watcher_config,
                get_target_log_path=self.get_llm_target_log_path,
                queue_live_prompt=self.queue_live_prompt_with_source,
                start_keyword_wait=self.start_keyword_wait,
                cancel_keyword_wait=self.cancel_keyword_wait,
            )
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def set_active_prompt_queue(self, prompt_queue: asyncio.Queue[PromptCommand]):
        self.active_prompt_queue = prompt_queue

    def clear_active_prompt_queue(self, prompt_queue: asyncio.Queue[PromptCommand]):
        if self.active_prompt_queue is prompt_queue:
            self.active_prompt_queue = None

    def set_active_conversation_log_path(self, log_path: Path):
        self.active_conversation_log_path = log_path
        self.last_conversation_log_path = log_path

    def clear_active_conversation_log_path(self, log_path: Path):
        if self.active_conversation_log_path == log_path:
            self.active_conversation_log_path = None
            self.last_conversation_log_path = log_path

    def get_llm_target_log_path(self) -> Optional[Path]:
        if self.active_conversation_log_path is not None:
            return self.active_conversation_log_path
        if self.last_conversation_log_path is not None and self.last_conversation_log_path.exists():
            return self.last_conversation_log_path
        log_dir = self.transcription_config.log_dir
        if not log_dir.exists():
            return None
        candidates = [path for path in log_dir.glob("*.log") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.name)

    def queue_live_prompt(self, prompt_text: str) -> bool:
        return self.queue_live_prompt_with_source(prompt_text, "manual")

    def queue_live_prompt_with_source(self, prompt_text: str, source: str = "manual") -> bool:
        prompt_queue = self.active_prompt_queue
        if prompt_queue is None:
            return False
        prompt_queue.put_nowait(PromptCommand(text=prompt_text, source=source))
        return True

    def start_keyword_wait(self, log_path: Path) -> bool:
        if self._llm_trigger_mode != "keyword":
            return False
        if self.active_conversation_log_path != log_path:
            return False
        if self.active_prompt_queue is None:
            return False
        if self._keyword_wait_active:
            return False
        self._keyword_wait_active = True
        self._keyword_wait_log_path = log_path
        return True

    def cancel_keyword_wait(self, log_path: Path) -> None:
        if self._keyword_wait_log_path == log_path:
            self._keyword_wait_active = False
            self._keyword_wait_log_path = None

    def finish_keyword_wait(self) -> None:
        self._keyword_wait_active = False
        self._keyword_wait_log_path = None

    def is_keyword_wait_active(self) -> bool:
        return self._keyword_wait_active

    def get_transcription_tag(self) -> str:
        if self._keyword_wait_active:
            return "user ignored"
        return "user"


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")
        session_id = uuid4().hex[:12]
        conversation_logger = ConversationLogger(self.transcription_config.log_dir, session_id)
        flush_phrases = ()
        if self._llm_trigger_mode == "keyword":
            flush_phrases = (self._llm_trigger_keyword,)
        model_text_logger = ModelTextLogger(conversation_logger, flush_phrases=flush_phrases)
        session_transcriber: SessionTranscriber | None = None
        self.set_active_conversation_log_path(conversation_logger.path)
        try:
            requested_voice_prompt_path = None
            voice_prompt_path = None
            if self.voice_prompt_dir is not None:
                voice_prompt_filename = request.query["voice_prompt"]
                if voice_prompt_filename is not None:
                    requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
                if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                    raise FileNotFoundError(
                        f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                    )
                voice_prompt_path = requested_voice_prompt_path

            if self.lm_gen.voice_prompt != voice_prompt_path:
                if voice_prompt_path.endswith(".pt"):
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(voice_prompt_path)

            initial_text_prompt = request.query["text_prompt"].strip()
            if self._llm_trigger_mode == "keyword":
                initial_text_prompt = build_keyword_mode_prompt(initial_text_prompt, self._llm_trigger_keyword)
            self.lm_gen.text_prompt_tokens = (
                self.text_tokenizer.encode(wrap_with_system_tags(initial_text_prompt))
                if initial_text_prompt
                else None
            )
            seed = int(request.query["seed"]) if "seed" in request.query else None
            session_prompt_queue: asyncio.Queue[PromptCommand] = asyncio.Queue()
            await conversation_logger.write_entry("initial_prompt", initial_text_prompt)

            async def recv_loop():
                nonlocal close
                try:
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.ERROR:
                            clog.log("error", f"{ws.exception()}")
                            break
                        if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                            break
                        if message.type != aiohttp.WSMsgType.BINARY:
                            clog.log("error", f"unexpected message type {message.type}")
                            continue
                        payload_message = message.data
                        if not isinstance(payload_message, bytes):
                            clog.log("error", f"unsupported message type {type(payload_message)}")
                            continue
                        if len(payload_message) == 0:
                            clog.log("warning", "empty message")
                            continue
                        kind = payload_message[0]
                        if kind == 1:
                            opus_reader.append_bytes(payload_message[1:])
                        else:
                            clog.log("warning", f"unknown message kind {kind}")
                finally:
                    close = True
                    clog.log("info", "connection closed")

            async def opus_loop():
                all_pcm_data = None
                effective_prompt_text = initial_text_prompt
                pending_prompt_commands: list[PromptCommand] = []
                prompt_interrupt_active = False
                boundary_streak = 0
                boundary_steps = 0

                def drain_prompt_queue():
                    nonlocal prompt_interrupt_active, all_pcm_data
                    while True:
                        try:
                            pending_prompt_commands.append(session_prompt_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    if pending_prompt_commands and not prompt_interrupt_active:
                        prompt_interrupt_active = True
                        all_pcm_data = None
                        _ = opus_reader.read_pcm()
                        clog.log("info", "queued live text prompt; interrupting user audio feed")

                async def emit_tokens(tokens: torch.Tensor):
                    text_token = tokens[0, 0, 0].item()
                    assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                    main_pcm = self.mimi.decode(tokens[:, 1:9])
                    _ = self.other_mimi.decode(tokens[:, 1:9])
                    main_pcm = main_pcm.cpu()
                    opus_writer.append_pcm(main_pcm[0, 0].numpy())
                    if text_token not in (0, TEXT_TOKEN_PAD):
                        piece = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                        piece = piece.replace("▁", " ")
                        await model_text_logger.append_piece(piece)
                        await ws.send_bytes(b"\x02" + bytes(piece, encoding="utf8"))
                    return text_token

                def build_effective_prompt(new_prompt_text: str) -> str:
                    if self.live_prompt_mode == "replace" or not effective_prompt_text:
                        return new_prompt_text
                    return f"{effective_prompt_text}\n{new_prompt_text}"

                def inject_prompt(prompt_text: str):
                    nonlocal effective_prompt_text
                    injected_prompt_text = apply_live_prompt_prefix(prompt_text, self.live_prompt_prefix)
                    effective_prompt_text = build_effective_prompt(injected_prompt_text)
                    prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(effective_prompt_text))
                    self.lm_gen.step_text_prompt_tokens(prompt_tokens)
                    self.lm_gen.step_audio_silence_frames(self.lm_gen.audio_silence_frame_cnt)
                    return injected_prompt_text

                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)
                    drain_prompt_queue()

                    if prompt_interrupt_active:
                        _ = opus_reader.read_pcm()
                        silence_tokens = self.lm_gen.step(
                            moshi_tokens=self.lm_gen._encode_zero_frame(),
                            text_token=self.lm_gen.zero_text_code,
                            input_tokens=self.lm_gen._encode_sine_frame(),
                        )
                        if silence_tokens is not None:
                            text_token = await emit_tokens(silence_tokens)
                            if text_token in (TEXT_TOKEN_EOS, TEXT_TOKEN_PAD):
                                boundary_streak += 1
                            else:
                                boundary_streak = 0
                        boundary_steps += 1
                        if boundary_streak >= LIVE_PROMPT_BOUNDARY_STREAK or boundary_steps >= LIVE_PROMPT_MAX_STEPS:
                            for prompt_command in pending_prompt_commands:
                                await model_text_logger.flush()
                                injected_prompt_text = inject_prompt(prompt_command.text)
                                await conversation_logger.write_entry("prompt", injected_prompt_text)
                                if prompt_command.source == "llm_keyword":
                                    self.finish_keyword_wait()
                                clog.log("info", f"applied live prompt: {injected_prompt_text}")
                            pending_prompt_commands.clear()
                            prompt_interrupt_active = False
                            boundary_streak = 0
                            boundary_steps = 0
                            all_pcm_data = None
                        continue

                    if self.is_keyword_wait_active():
                        all_pcm_data = None
                        pcm = opus_reader.read_pcm()
                        if pcm.shape[-1] != 0 and session_transcriber is not None:
                            session_transcriber.submit_pcm(pcm)
                        continue

                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] == 0:
                        continue
                    if session_transcriber is not None:
                        session_transcriber.submit_pcm(pcm)
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        _ = self.other_mimi.encode(chunk)
                        for c in range(codes.shape[-1]):
                            drain_prompt_queue()
                            if prompt_interrupt_active:
                                break
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            await emit_tokens(tokens)
                        if prompt_interrupt_active or all_pcm_data is None:
                            break

            async def send_loop():
                while True:
                    if close:
                        return
                    await asyncio.sleep(0.001)
                    msg = opus_writer.read_bytes()
                    if len(msg) > 0:
                        await ws.send_bytes(b"\x01" + msg)

            clog.log("info", "accepted connection")
            if initial_text_prompt:
                clog.log("info", f"text prompt: {initial_text_prompt}")
            if len(request.query["voice_prompt"]) > 0:
                clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
            close = False
            async with self.lock:
                if seed is not None and seed != -1:
                    seed_all(seed)

                opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
                opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
                self.mimi.reset_streaming()
                self.other_mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                if self.transcription_service is not None:
                    session_transcriber = self.transcription_service.create_session(
                        conversation_logger=conversation_logger,
                        sample_rate=self.mimi.sample_rate,
                        get_log_tag=self.get_transcription_tag,
                    )
                    session_transcriber.start()

                async def is_alive():
                    if close or ws.closed:
                        return False
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            return False
                    except asyncio.TimeoutError:
                        return True
                    except aiohttp.ClientConnectionError:
                        return False
                    return True

                await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
                self.mimi.reset_streaming()
                clog.log("info", "done with system prompts")
                if await is_alive():
                    await ws.send_bytes(b"\x00")
                    clog.log("info", "sent handshake bytes")
                    self.set_active_prompt_queue(session_prompt_queue)
                    try:
                        tasks = [
                            asyncio.create_task(recv_loop()),
                            asyncio.create_task(opus_loop()),
                            asyncio.create_task(send_loop()),
                        ]
                        _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        await ws.close()
                        clog.log("info", "session closed")
                    finally:
                        self.finish_keyword_wait()
                        self.clear_active_prompt_queue(session_prompt_queue)
            clog.log("info", "done with connection")
            return ws
        finally:
            self.finish_keyword_wait()
            self.clear_active_conversation_log_path(conversation_logger.path)
            await model_text_logger.flush()
            if session_transcriber is not None:
                await session_transcriber.close()
            await conversation_logger.close()


async def stdin_prompt_loop(state: ServerState):
    logger.info("live prompt stdin enabled; enter one prompt per line")
    while True:
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"stdin prompt loop failed: {exc}")
            return

        if line == "":
            logger.info("stdin closed; stopping live prompt loop")
            return

        prompt_text = line.strip()
        if not prompt_text:
            continue

        if state.queue_live_prompt_with_source(prompt_text, "manual"):
            logger.info(f"queued live prompt: {prompt_text}")
        else:
            logger.warning("ignored live prompt because no active session is running")


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        repo_root = Path(__file__).resolve().parents[2]
        local_dist = repo_root / "client" / "dist"
        if local_dist.exists():
            logger.info(f"using local frontend build at {local_dist}")
            return str(local_dist)
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--live-prompt-stdin",
        action="store_true",
        help="Read one live prompt per line from stdin and inject it into the active session.",
    )
    parser.add_argument(
        "--live-prompt-mode",
        choices=("append", "replace"),
        default="replace",
        help="How stdin live prompts combine with the current session prompt.",
    )
    parser.add_argument(
        "--live-prompt-prefix",
        type=str,
        default="[SYSTEM PROMPT]:",
        help="Prefix added to each injected live prompt before it is sent to the model.",
    )
    parser.add_argument(
        "--enable-transcription",
        action="store_true",
        help="Run background user speech-to-text transcription and persist per-session conversation logs.",
    )
    parser.add_argument(
        "--transcription-model-id",
        type=str,
        default="distil-whisper/distil-large-v3",
        help="Hugging Face model id to use for background user transcription.",
    )
    parser.add_argument(
        "--conversation-log-dir",
        type=str,
        default="./logs/conversations",
        help="Directory where per-session conversation logs are written.",
    )
    parser.add_argument(
        "--transcription-chunk-seconds",
        type=float,
        default=6.0,
        help="User audio chunk size in seconds for each background transcription pass.",
    )
    parser.add_argument(
        "--transcription-overlap-seconds",
        type=float,
        default=1.5,
        help="Overlap between consecutive transcription chunks in seconds.",
    )
    parser.add_argument(
        "--llm-log-watcher",
        action="store_true",
        help="Watch conversation logs, send them to an OpenAI model, and inject the result as a live prompt.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-nano",
        help="OpenAI model used by the conversation log watcher.",
    )
    parser.add_argument(
        "--llm-system-prompt-file",
        type=str,
        default=str(Path(__file__).with_name("llm_sys_prompt.txt")),
        help="Path to the text file used as the system prompt for the log watcher.",
    )
    parser.add_argument(
        "--llm-trigger-mode",
        choices=("user", "any", "keyword"),
        default="user",
        help="Which new log lines should trigger an LLM call.",
    )
    parser.add_argument(
        "--llm-trigger-keyword",
        type=str,
        default="my supervisor",
        help="Case-insensitive phrase that triggers supervisor lookup in --llm-trigger-mode keyword.",
    )
    parser.add_argument(
        "--llm-payload-mode",
        choices=("rolling", "full"),
        default="rolling",
        help="Whether to send a rolling log window or the full log file to the LLM.",
    )
    parser.add_argument(
        "--llm-rolling-lines",
        type=int,
        default=15,
        help="Number of latest tagged log lines to send when --llm-payload-mode rolling is used.",
    )
    parser.add_argument(
        "--llm-injection-template",
        type=str,
        default="",
        help="Optional wrapper template applied before live prompt injection. Use {prompt} as the placeholder.",
    )
    parser.add_argument(
        "--llm-poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval in seconds for the conversation log watcher.",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    if args.transcription_chunk_seconds <= 0:
        raise ValueError("--transcription-chunk-seconds must be > 0")
    if args.transcription_overlap_seconds < 0:
        raise ValueError("--transcription-overlap-seconds must be >= 0")
    if args.llm_rolling_lines <= 0:
        raise ValueError("--llm-rolling-lines must be > 0")
    if args.llm_poll_seconds <= 0:
        raise ValueError("--llm-poll-seconds must be > 0")
    if args.llm_injection_template and "{prompt}" not in args.llm_injection_template:
        raise ValueError("--llm-injection-template must include the {prompt} placeholder")
    if args.llm_trigger_mode == "keyword" and not args.llm_trigger_keyword.strip():
        raise ValueError("--llm-trigger-keyword must not be empty when --llm-trigger-mode keyword is used")
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)
    if args.enable_transcription:
        logger.info(
            "user transcription enabled: model=%s log_dir=%s chunk=%.2fs overlap=%.2fs",
            args.transcription_model_id,
            args.conversation_log_dir,
            args.transcription_chunk_seconds,
            args.transcription_overlap_seconds,
        )
    if args.llm_log_watcher:
        logger.info(
            "llm log watcher enabled: model=%s system_prompt=%s trigger=%s keyword=%s payload=%s poll=%.2fs",
            args.llm_model,
            args.llm_system_prompt_file,
            args.llm_trigger_mode,
            args.llm_trigger_keyword,
            args.llm_payload_mode,
            args.llm_poll_seconds,
        )

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        live_prompt_mode=args.live_prompt_mode,
        live_prompt_prefix=args.live_prompt_prefix,
        transcription_config=TranscriptionConfig(
            enabled=args.enable_transcription,
            model_id=args.transcription_model_id,
            chunk_seconds=args.transcription_chunk_seconds,
            overlap_seconds=args.transcription_overlap_seconds,
            log_dir=Path(args.conversation_log_dir),
        ),
        llm_watcher_config=LLMWatcherConfig(
            enabled=args.llm_log_watcher,
            model=args.llm_model,
            system_prompt_file=Path(args.llm_system_prompt_file),
            trigger_mode=args.llm_trigger_mode,
            trigger_keyword=args.llm_trigger_keyword.strip(),
            payload_mode=args.llm_payload_mode,
            rolling_lines=args.llm_rolling_lines,
            injection_template=args.llm_injection_template,
            poll_seconds=args.llm_poll_seconds,
            log_dir=Path(args.conversation_log_dir),
        ),
    )
    if state.llm_watcher is not None:
        state.llm_watcher.startup()
    logger.info("warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    stdin_task: Optional[asyncio.Task] = None
    llm_watcher_task: Optional[asyncio.Task] = None

    async def on_startup(_app):
        nonlocal stdin_task, llm_watcher_task
        if args.live_prompt_stdin:
            stdin_task = asyncio.create_task(stdin_prompt_loop(state))
        if state.llm_watcher is not None:
            llm_watcher_task = asyncio.create_task(state.llm_watcher.run())

    async def on_cleanup(_app):
        if stdin_task is not None:
            stdin_task.cancel()
            with suppress(asyncio.CancelledError):
                await stdin_task
        if llm_watcher_task is not None:
            llm_watcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await llm_watcher_task
        if state.transcription_service is not None:
            state.transcription_service.shutdown()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
