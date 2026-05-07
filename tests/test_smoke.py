"""Smoke tests — agent imports, FAQ corpus integrity, KB lookups."""
from __future__ import annotations

from typing import get_type_hints


def test_agent_module_imports():
    import agent  # noqa: F401
    import knowledge.faqs  # noqa: F401
    import knowledge.kb  # noqa: F401


def test_faq_count_is_15():
    from knowledge.faqs import FAQS
    assert len(FAQS) == 15


def test_faq_entries_well_formed():
    from knowledge.faqs import FAQS
    for slug, entry in FAQS.items():
        assert slug and entry["question"].strip() and entry["answer"].strip()


def test_info_agent_has_no_function_tools():
    """KB is inlined into the system prompt — InfoAgent must expose ZERO
    @function_tool methods so the LLM answers in a single turn instead
    of round-tripping through tool calls. Tool calls were the root cause
    of multiple Soniox TTS quirks (empty-turn audio, latency)."""
    from agent import InfoAgent

    tool_names = []
    for attr_name in dir(InfoAgent):
        attr = getattr(InfoAgent, attr_name, None)
        if callable(attr) and getattr(attr, "__wrapped__", None) is not None:
            tool_names.append(attr_name)

    assert tool_names == [], (
        f"InfoAgent must have no @function_tool methods, got: {tool_names}"
    )


def test_kb_text_inlined_into_instructions():
    """Every FAQ question and every deadline scenario must be present in
    the system prompt verbatim — that's the whole point of inlining."""
    from agent import INSTRUCTIONS
    from knowledge.faqs import FAQS
    from knowledge.kb import DEADLINES, CONTACTS, SENIUNIJOS_INFO

    for slug, entry in FAQS.items():
        assert entry["question"] in INSTRUCTIONS, f"FAQ {slug!r} question missing from prompt"
        assert entry["answer"] in INSTRUCTIONS, f"FAQ {slug!r} answer missing from prompt"

    for slug, deadline in DEADLINES.items():
        assert deadline in INSTRUCTIONS, f"Deadline {slug!r} missing from prompt"

    assert CONTACTS["savivaldybes_telefonas"] in INSTRUCTIONS
    assert SENIUNIJOS_INFO in INSTRUCTIONS


def test_kb_text_size_under_4k_tokens():
    """Prompt-cache friendliness gate: keep the inlined KB under ~4k
    tokens (~16 KB) so the cached prefix stays cheap."""
    from agent import KB_TEXT
    assert len(KB_TEXT) < 16000, f"KB_TEXT is {len(KB_TEXT)} chars — over the 16 KB ceiling"


def test_tts_sanitizer_strips_markdown_and_brackets():
    """The Soniox TTS sanitizer must strip markdown links, bracket tags,
    bold/italic, and replace bare URLs with TTS-friendly spelling."""
    from agent import _sanitize_for_tts

    raw = "Apsilankykite [www.epaslaugos.lt](http://www.epaslaugos.lt) puslapyje."
    out = _sanitize_for_tts(raw)
    assert "[" not in out and "]" not in out and "(" not in out
    assert "epaslaugos taškas l t" in out

    # Bracket audio tags — stripped, surrounding text preserved.
    out = _sanitize_for_tts("[warmly] Sveiki, [thoughtfully] klausau.")
    assert "[" not in out and "warmly" not in out and "thoughtfully" not in out
    assert "Sveiki," in out and "klausau." in out

    # Markdown emphasis — markers stripped, text kept.
    assert _sanitize_for_tts("Tai **labai svarbu** ir *būtina*.") == "Tai labai svarbu ir būtina."

    # Bare URL — spelled out.
    out = _sanitize_for_tts("Užpildyti formą galite www.epaslaugos.lt portale.")
    assert "www" not in out and "epaslaugos taškas l t" in out

    # Plain text — unchanged.
    assert _sanitize_for_tts("Sveiki, padėsiu dėl deklaravimo.") == "Sveiki, padėsiu dėl deklaravimo."


def test_sanitizer_preserves_prosody_chunks():
    """Punctuation and leading whitespace must reach Soniox TTS — they
    drive prosody. Dropping them was the cause of the 'concatenated
    monotone' speech bug."""
    from agent import _sanitize_for_tts

    # Punctuation chunks pass through unchanged.
    assert _sanitize_for_tts(",") == ","
    assert _sanitize_for_tts(".") == "."
    assert _sanitize_for_tts("?") == "?"
    assert _sanitize_for_tts("!") == "!"

    # Leading space preserved (separates words across chunks).
    assert _sanitize_for_tts(" jei") == " jei"
    assert _sanitize_for_tts(" išvykstate") == " išvykstate"

    # Trailing newline preserved (sentence boundary).
    assert _sanitize_for_tts("Lietuvos.\n") == "Lietuvos.\n"


def test_kb_lookups_work():
    from knowledge.faqs import get_faq, faq_index
    from knowledge.kb import DEADLINES, REQUIRED_DOCS, CONTACTS, SENIUNIJOS_INFO

    assert get_faq("kaip_deklaruoti")["answer"]
    assert get_faq("nope") is None
    assert "kaip_deklaruoti" in faq_index()
    assert "septynias darbo dienas" in DEADLINES["isvykimas_iz_lt"]
    assert any("nuomos" in d for d in REQUIRED_DOCS["persikraustymas_lt"])
    assert CONTACTS["savivaldybes_telefonas"].startswith("+370")
    assert "Šnipiškių" in SENIUNIJOS_INFO
