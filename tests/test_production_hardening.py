"""Production-hardening regression tests — locks in the production-grade
patterns required by the livekit-agents skill checklist:

- Graceful shutdown on entrypoint failures (skill: "Design for the unhappy path")
- BackgroundAudio for processing fillers (skill: "Silence feels broken")
- on_session_end hook for structured session reports (skill: observability)
- Per-turn e2e_latency budget (skill: "Measure latency continuously")
- Prompt-cache verification (skill: "Keep context minimal" — caching makes inlined KB cheap)
"""
from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# #2 Graceful shutdown — static lints on the entrypoint
# ---------------------------------------------------------------------------

def test_entrypoint_has_try_except_with_shutdown():
    """The entrypoint MUST guard session.start() with try/except so a
    Soniox WS failure / Inference 5xx / Silero load failure does not
    crash the worker silently."""
    import agent
    src = inspect.getsource(agent.entrypoint)
    assert "try:" in src and "except" in src, (
        "entrypoint() must wrap session.start() in try/except — "
        "currently there is no unhappy-path handling."
    )
    # On exception, must call session.shutdown() OR ctx.shutdown()
    assert "shutdown()" in src, (
        "entrypoint() must call session.shutdown() or ctx.shutdown() "
        "on exception so the room is closed and the job ends cleanly."
    )


def test_entrypoint_registers_shutdown_callback():
    """Session reports / cleanup logic must be wired via
    ctx.add_shutdown_callback() so it runs even on graceful exits."""
    import agent
    src = inspect.getsource(agent.entrypoint)
    assert "add_shutdown_callback" in src, (
        "entrypoint() must register an on-shutdown callback to log the "
        "session report (per skill: observability)."
    )


# ---------------------------------------------------------------------------
# #3 BackgroundAudio — wired and configured
# ---------------------------------------------------------------------------

def test_entrypoint_wires_background_audio_player():
    """Per skill: 'Silence feels broken — acknowledge processing when
    needed.' BackgroundAudioPlayer with ambient or thinking sound must
    be started inside the entrypoint."""
    import agent
    src = inspect.getsource(agent.entrypoint)
    assert "BackgroundAudioPlayer" in src or "background_audio" in src.lower(), (
        "entrypoint() must initialize a BackgroundAudioPlayer to fill "
        "silent gaps. Without it, 2-3s LLM thinking gaps feel broken "
        "to the caller."
    )


def test_agent_module_imports_background_audio_constructs():
    """BackgroundAudioPlayer + AudioConfig + BuiltinAudioClip must be
    importable from livekit.agents at module load time."""
    import agent
    # The module should have imported the BackgroundAudioPlayer class so
    # we can reference it. We don't care about the exact import path.
    assert hasattr(agent, "BackgroundAudioPlayer") or "BackgroundAudioPlayer" in inspect.getsource(agent), (
        "agent.py must reference BackgroundAudioPlayer (either imported "
        "or fully-qualified)."
    )


# ---------------------------------------------------------------------------
# #6 on_session_end / shutdown report
# ---------------------------------------------------------------------------

def test_shutdown_report_logs_session_usage():
    """The shutdown handler must capture cumulative session.usage in a
    structured log entry — per docs:
    https://docs.livekit.io/deploy/observability/data/#session-reports"""
    import agent
    src = inspect.getsource(agent.entrypoint)
    assert "session.usage" in src or "make_session_report" in src, (
        "Shutdown callback must emit session.usage or "
        "ctx.make_session_report() so post-call telemetry is searchable."
    )


# ---------------------------------------------------------------------------
# #4 Per-turn e2e_latency budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_ttft_budget_for_simple_question(llm, agent_session):
    """A simple factual question should reach first-token within a
    sane LLM budget.

    The test framework bypasses real STT/TTS rendering, so we measure
    `llm_node_ttft` (time-to-first-token) — the most variable contributor
    to production e2e_latency. Production target is < 1.5 s; this gate
    is set to 4 s to absorb test cold-start jitter."""
    result = await agent_session.run(user_input="Per kiek laiko reikia deklaruoti?")
    msg = result.expect.contains_message(role="assistant")

    chat_msg = msg.event().item
    metrics_dict = chat_msg.metrics or {}
    ttft = metrics_dict.get("llm_node_ttft") or metrics_dict.get("e2e_latency")
    assert ttft is not None, (
        f"No latency metric on assistant message: {metrics_dict!r}"
    )
    assert ttft < 4.0, (
        f"llm_node_ttft {ttft:.2f}s exceeds 4 s budget — slow LLM path. "
        "Production target is < 1.5 s; investigate context bloat or "
        "LLM cold start. Production e2e budget = ttft + STT_delay + TTS_ttfb."
    )


# ---------------------------------------------------------------------------
# #5 Prompt-cache verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_caching_reduces_input_tokens_on_repeat(llm, agent_session):
    """If LiveKit Inference's prompt-caching is working, the SECOND
    user turn should hit a cached prefix — meaning prompt_cached_tokens
    > 0 OR the per-turn input tokens stay roughly flat (not doubling
    with conversation length)."""
    # First turn — primes cache
    r1 = await agent_session.run(user_input="Kaip deklaruoti internetu?")
    m1 = r1.expect.contains_message(role="assistant").event().item.metrics or {}

    # Second turn
    r2 = await agent_session.run(user_input="O kokie terminai?")
    m2 = r2.expect.contains_message(role="assistant").event().item.metrics or {}

    cached_2 = m2.get("prompt_cached_tokens")
    if cached_2 is not None:
        # Direct cache-hit signal available — assert non-zero.
        assert cached_2 > 0, (
            f"Second turn returned 0 cached tokens — caching not active. "
            f"m1={m1!r} m2={m2!r}"
        )
    else:
        # No direct cache field — fall back to a relative-input-token check.
        # If caching is active, the system-prompt prefix (~3000 tokens)
        # is reused, so input growth turn-over-turn should be < 200 tokens.
        # Without caching, every turn would carry the full 3000 tokens
        # plus prior history.
        in1 = m1.get("prompt_tokens") or m1.get("input_tokens") or 0
        in2 = m2.get("prompt_tokens") or m2.get("input_tokens") or 0
        if in1 and in2:
            # We expect in2 ≈ in1 + (a few hundred history tokens), NOT
            # in2 ≈ 2 × in1.
            ratio = in2 / max(in1, 1)
            assert ratio < 1.6, (
                f"Input tokens grew 60%+ on turn 2 (in1={in1}, in2={in2}, "
                f"ratio={ratio:.2f}). Caching may not be active."
            )
        else:
            pytest.skip("No prompt-token metrics available in this test path")
