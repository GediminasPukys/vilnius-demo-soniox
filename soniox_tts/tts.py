from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

logger = logging.getLogger("soniox.tts")

REST_URL = "https://tts-rt.soniox.com/tts"
WS_URL = "wss://tts-rt.soniox.com/tts-websocket"

DEFAULT_MODEL = "tts-rt-v1"
DEFAULT_VOICE = "Adrian"
DEFAULT_LANGUAGE = "en"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1

AudioFormat = Literal["pcm_s16le", "pcm_f32le", "pcm_mulaw", "pcm_alaw", "wav", "mp3", "opus", "flac", "aac"]


@dataclass
class TTSOptions:
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    audio_format: AudioFormat = "pcm_s16le"
    sample_rate: int = DEFAULT_SAMPLE_RATE


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        resolved_key = api_key if is_given(api_key) else os.environ.get("SONIOX_API_KEY")
        if not resolved_key:
            raise ValueError("SONIOX_API_KEY is required (pass api_key= or set env var)")
        self._api_key = resolved_key

        self._opts = TTSOptions(
            model=model if is_given(model) else DEFAULT_MODEL,
            voice=voice if is_given(voice) else DEFAULT_VOICE,
            language=language if is_given(language) else DEFAULT_LANGUAGE,
            audio_format="pcm_s16le",
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Soniox"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        payload = {
            "model": self._opts.model,
            "language": self._opts.language,
            "voice": self._opts.voice,
            "audio_format": self._opts.audio_format,
            "sample_rate": self._opts.sample_rate,
            "text": self.input_text,
        }
        headers = {
            "Authorization": f"Bearer {self._tts._api_key}",
            "Content-Type": "application/json",
        }

        session = self._tts._ensure_session()
        try:
            async with session.post(
                REST_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise APIStatusError(
                        f"Soniox TTS error: {body}",
                        status_code=resp.status,
                        request_id=request_id,
                        body=body,
                    )
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        output_emitter.push(chunk)

            output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = str(uuid.uuid4())
        stream_id = f"stream-{uuid.uuid4().hex[:12]}"

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        session = self._tts._ensure_session()
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(WS_URL),
                timeout=self._conn_options.timeout,
            )

            config = {
                "api_key": self._tts._api_key,
                "stream_id": stream_id,
                "model": self._opts.model,
                "language": self._opts.language,
                "voice": self._opts.voice,
                "audio_format": self._opts.audio_format,
                "sample_rate": self._opts.sample_rate,
            }
            await ws.send_str(json.dumps(config))

            send_task = asyncio.create_task(self._send_text_task(ws, stream_id), name="soniox_tts_send")
            recv_task = asyncio.create_task(self._recv_audio_task(ws, output_emitter, stream_id), name="soniox_tts_recv")
            keepalive_task = asyncio.create_task(self._keepalive_task(ws), name="soniox_tts_keepalive")

            try:
                # Wait for send + recv to finish; keepalive is cancelled when they do.
                await asyncio.gather(send_task, recv_task)
            except Exception:
                for t in (send_task, recv_task, keepalive_task):
                    if not t.done():
                        t.cancel()
                raise
            finally:
                if not keepalive_task.done():
                    keepalive_task.cancel()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None and not ws.closed:
                await ws.close()

    async def _keepalive_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> None:
        """Send ``{"keep_alive": true}`` every 20 s.

        Soniox closes idle WebSocket connections after 20–30 s. Without this,
        a quiet user-pause mid-call can cause the next utterance to be
        rendered on a half-closed socket — which manifests as garbled or
        truncated audio. Mirrors Pipecat's keepalive cadence.
        """
        try:
            while not ws.closed:
                await asyncio.sleep(20)
                if ws.closed:
                    return
                try:
                    await ws.send_str(json.dumps({"keep_alive": True}))
                except (aiohttp.ClientError, ConnectionResetError):
                    return
        except asyncio.CancelledError:
            return

    async def _send_text_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        stream_id: str,
    ) -> None:
        chunks_sent = 0
        accumulated = ""
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                logger.info(
                    f"[TTS→Soniox] flush stream_id={stream_id} "
                    f"chunks_sent={chunks_sent} total_text={accumulated!r}"
                )
                await ws.send_str(json.dumps({
                    "stream_id": stream_id,
                    "text": "",
                    "text_end": True,
                }))
                return
            if not data:
                continue
            self._mark_started()
            chunks_sent += 1
            accumulated += data
            logger.info(
                f"[TTS→Soniox] chunk #{chunks_sent} stream_id={stream_id} "
                f"text={data!r}"
            )
            await ws.send_str(json.dumps({
                "stream_id": stream_id,
                "text": data,
                "text_end": False,
            }))

        # Channel closed without flush
        logger.info(
            f"[TTS→Soniox] close stream_id={stream_id} "
            f"chunks_sent={chunks_sent} total_text={accumulated!r}"
        )
        await ws.send_str(json.dumps({
            "stream_id": stream_id,
            "text": "",
            "text_end": True,
        }))

    async def _recv_audio_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        stream_id: str,
    ) -> None:
        msgs_total = 0
        audio_chunks_total = 0
        audio_bytes_total = 0
        try:
            async for msg in ws:
                msgs_total += 1
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception as e:
                        logger.warning(f"[Soniox→TTS] non-JSON text frame stream_id={stream_id}: {msg.data!r} err={e!r}")
                        continue

                    msg_stream_id = data.get("stream_id")
                    if msg_stream_id != stream_id:
                        logger.debug(
                            f"[Soniox→TTS] msg for other stream_id={msg_stream_id} (mine={stream_id}); ignoring"
                        )
                        continue

                    if "error_code" in data:
                        logger.error(
                            f"[Soniox→TTS] ERROR stream_id={stream_id} "
                            f"code={data.get('error_code')} msg={data.get('error_message')!r} "
                            f"raw={msg.data!r}"
                        )
                        raise APIStatusError(
                            data.get("error_message", "Soniox TTS error"),
                            status_code=data.get("error_code", 500),
                            request_id=stream_id,
                            body=msg.data,
                        )

                    audio_b64 = data.get("audio")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_chunks_total += 1
                        audio_bytes_total += len(audio_bytes)
                        if audio_chunks_total <= 3 or audio_chunks_total % 20 == 0:
                            logger.info(
                                f"[Soniox→TTS] audio chunk #{audio_chunks_total} "
                                f"stream_id={stream_id} size={len(audio_bytes)}B "
                                f"total={audio_bytes_total}B"
                            )
                        output_emitter.push(audio_bytes)

                    if data.get("terminated"):
                        logger.info(
                            f"[Soniox→TTS] TERMINATED stream_id={stream_id} "
                            f"msgs={msgs_total} audio_chunks={audio_chunks_total} "
                            f"audio_bytes={audio_bytes_total}"
                        )
                        output_emitter.end_segment()
                        return
                    if data.get("audio_end"):
                        logger.info(
                            f"[Soniox→TTS] audio_end stream_id={stream_id} "
                            f"audio_chunks={audio_chunks_total} bytes={audio_bytes_total}"
                        )
                        # final audio for this segment delivered; wait for terminated
                        continue
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    audio_chunks_total += 1
                    audio_bytes_total += len(msg.data)
                    logger.debug(
                        f"[Soniox→TTS] binary frame #{audio_chunks_total} "
                        f"size={len(msg.data)}B"
                    )
                    output_emitter.push(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(
                        f"[Soniox→TTS] ws {msg.type.name} stream_id={stream_id} "
                        f"msgs={msgs_total} audio_chunks={audio_chunks_total}"
                    )
                    return
        finally:
            logger.info(
                f"[Soniox→TTS] recv loop done stream_id={stream_id} "
                f"msgs={msgs_total} audio_chunks={audio_chunks_total} "
                f"audio_bytes={audio_bytes_total}"
            )
