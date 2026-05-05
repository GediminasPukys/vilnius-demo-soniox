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


def test_function_tools_have_resolvable_type_hints():
    from agent import InfoAgent

    failures = []
    for attr_name in dir(InfoAgent):
        attr = getattr(InfoAgent, attr_name, None)
        wrapped = getattr(attr, "__wrapped__", None) if callable(attr) else None
        if wrapped is None:
            continue
        try:
            get_type_hints(wrapped, include_extras=True)
        except Exception as e:  # noqa: BLE001
            failures.append((attr_name, repr(e)))

    assert not failures, f"Tool type hints failed: {failures}"


def test_tts_sanitizer_strips_markdown_and_brackets():
    """The Soniox TTS sanitizer must strip markdown links, bracket tags,
    bold/italic, and replace bare URLs with TTS-friendly spelling."""
    from agent import _sanitize_for_tts

    raw = "Apsilankykite [www.epaslaugos.lt](http://www.epaslaugos.lt) puslapyje."
    out = _sanitize_for_tts(raw)
    assert "[" not in out and "]" not in out and "(" not in out
    assert "epaslaugos taškas l t" in out

    # Bracket audio tags — stripped, surrounding text preserved.
    assert _sanitize_for_tts("[warmly] Sveiki, [thoughtfully] klausau.") == "Sveiki, klausau."

    # Markdown emphasis — markers stripped, text kept.
    assert _sanitize_for_tts("Tai **labai svarbu** ir *būtina*.") == "Tai labai svarbu ir būtina."

    # Bare URL — spelled out.
    out = _sanitize_for_tts("Užpildyti formą galite www.epaslaugos.lt portale.")
    assert "www" not in out and "epaslaugos taškas l t" in out

    # Plain text — unchanged.
    assert _sanitize_for_tts("Sveiki, padėsiu dėl deklaravimo.") == "Sveiki, padėsiu dėl deklaravimo."


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
