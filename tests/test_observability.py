"""Observability-integration regression tests.

The voice-agent-observability platform requires three things, in order:
  1. `init_observability(...)` called at module-import time.
  2. `obs.attach(ctx, session)` called BEFORE `session.start()`.
  3. `OBS_INGEST_URL` / `OBS_API_KEY` env vars present (or empty for
     disabled mode).

These contracts are easy to break silently — the SDK fails closed
(disabled mode), which means a regression hides itself. These tests
lock the wiring into place.
"""
from __future__ import annotations

import inspect


def test_agent_imports_obs_sdk():
    """The SDK must be imported at module level — both `init_observability`
    and `obs` must be reachable."""
    import agent
    assert hasattr(agent, "init_observability"), (
        "agent.py must import init_observability from voice_agent_observability"
    )
    assert hasattr(agent, "obs"), (
        "agent.py must import obs from voice_agent_observability"
    )


def test_init_observability_called_at_import_time():
    """init_observability(...) must be called at module level (NOT
    inside entrypoint), so the log-forwarding handler attaches before
    any session can start. Verified by source inspection."""
    import agent
    src = inspect.getsource(agent)
    # Find the call site at module level — it must NOT be indented
    # inside entrypoint().
    lines = src.split("\n")
    init_calls = [
        (i, line) for i, line in enumerate(lines)
        if line.strip().startswith("init_observability(")
    ]
    assert init_calls, "init_observability(...) must be called somewhere in agent.py"
    # Module-level call: the line must start at column 0 (no indent).
    module_level = [line for _, line in init_calls if not line[:1].isspace()]
    assert module_level, (
        "init_observability(...) must be at module level (column 0). "
        "Calling it inside entrypoint() means log forwarding misses "
        "anything emitted before the first job starts."
    )


def test_init_observability_uses_correct_slug():
    """The agent_slug passed to init_observability must match the
    slug minted in the obs backend ('vilnius-soniox')."""
    import agent
    src = inspect.getsource(agent)
    assert 'agent_slug="vilnius-soniox"' in src or "agent_slug='vilnius-soniox'" in src, (
        "init_observability must be called with agent_slug='vilnius-soniox' "
        "to match the API key minted in the obs backend."
    )


def test_obs_attach_called_before_session_start():
    """obs.attach(ctx, session) must appear BEFORE session.start() in
    the entrypoint. Calling it after has no effect (SDK registers
    listeners on the session at attach time)."""
    import agent
    src = inspect.getsource(agent.entrypoint)

    attach_idx = src.find("obs.attach(")
    # Match the actual call site, not docstring references — `await ` prefix
    # is what distinguishes the call from a free-form mention.
    start_idx = src.find("await session.start(")

    assert attach_idx >= 0, "obs.attach(...) must be called in entrypoint"
    assert start_idx >= 0, "session.start(...) must be called in entrypoint"
    assert attach_idx < start_idx, (
        f"obs.attach must come BEFORE session.start. Got attach at "
        f"position {attach_idx}, session.start at {start_idx}."
    )


def test_obs_attach_signature():
    """obs.attach must be called with both ctx and session. Wrong arg
    order is the most common silent failure."""
    import agent
    src = inspect.getsource(agent.entrypoint)
    # Either positional or named — both are fine, but ctx before session.
    has_correct_call = (
        "obs.attach(ctx, session)" in src
        or "obs.attach(ctx=ctx, session=session)" in src
    )
    assert has_correct_call, (
        "obs.attach must be called as obs.attach(ctx, session). "
        "Other orderings or missing args are silent failures."
    )
