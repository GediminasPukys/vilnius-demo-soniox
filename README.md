# Vilnius Savivaldybė LiveKit Demo — Soniox STT + Soniox TTS

Simplified LiveKit voice agent for Vilniaus miesto savivaldybė that answers
residents' questions about residence declaration. End-to-end Soniox stack:
**Soniox STT** for input + **Soniox TTS** for voice output (Lithuanian).

The Soniox TTS plugin is not in the official LiveKit plugin catalog yet —
this project bundles a local `sioniox/` package (ported from the user's
`soniox_livekit_agent` reference project) that wraps Soniox's TTS REST +
WebSocket API into LiveKit's `tts.TTS` interface.

## Stack

| Component | Provider |
|---|---|
| STT | Soniox `stt-rt-v4`, Lithuanian strict |
| LLM | `openai/gpt-5.3-chat-latest` via LiveKit Inference |
| TTS | Soniox `tts-rt-v1-preview` (local `sioniox/` plugin), voice "Adrian", `language="lt"` |
| Turn detection | Soniox STT endpointing only — **no turn-detector model** |
| VAD | Silero |

## Layout

```
agent.py            # AgentSession + InfoAgent + tools + STT context (one file)
sioniox/            # Local Soniox TTS LiveKit plugin (REST + WebSocket)
  __init__.py
  tts.py
knowledge/
  faqs.py           # 15 FAQ Q&A from paslaugos.vilnius.lt
  kb.py             # deadlines, documents, contacts
tests/
  conftest.py
  test_smoke.py
  test_faq_behavioral.py
```

## Run

```shell
cp .env.example .env   # fill in LIVEKIT_*, SONIOX_API_KEY
uv sync
uv run python agent.py download-files
uv run python agent.py dev
```

## Test

```shell
uv run pytest tests/test_smoke.py -v          # fast, no LLM
uv run pytest tests/test_faq_behavioral.py -v # 19 LLM-judged behavioral tests
```

## Notes

- Soniox TTS uses ONE API key for both STT and TTS (`SONIOX_API_KEY`).
- The Soniox TTS plugin is bundled in this repo (`sioniox/`). When/if it
  gets adopted into the official `livekit-plugins-soniox` package,
  swap `import sioniox` → `from livekit.plugins.soniox import TTS`.
- Soniox TTS does not interpret inline audio tags like `[warmly]` —
  the system prompt forbids them for this variant.
