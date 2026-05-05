"""Test fixtures for the Vilnius (ElevenLabs v3) agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env")

os.environ.setdefault("SONIOX_API_KEY", "test")
os.environ.setdefault("LIVEKIT_URL", "wss://test.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test")
os.environ.setdefault("LIVEKIT_API_SECRET", "test")


def _has_real_livekit_creds() -> bool:
    key = os.environ.get("LIVEKIT_API_KEY", "")
    secret = os.environ.get("LIVEKIT_API_SECRET", "")
    return bool(key) and key != "test" and bool(secret) and secret != "test"


@pytest.fixture
async def llm():
    if not _has_real_livekit_creds():
        pytest.skip("LIVEKIT creds not set — skipping behavioral test")
    from livekit.agents import inference
    async with inference.LLM(model="openai/gpt-5.3-chat-latest") as instance:
        yield instance


@pytest.fixture
async def agent_session(llm):
    from livekit.agents.voice import AgentSession
    from agent import CallState, InfoAgent

    async with AgentSession[CallState](userdata=CallState(), llm=llm) as session:
        await session.start(InfoAgent())
        yield session
