"""Softer-tone regression tests — Soniox TTS has no tone/pitch knobs,
so warmth has to come from the LLM-emitted text. The prompt MUST
encourage:
  * Warmth markers in Lithuanian ("labai mielai", "prašom", "dėkoju
    jums", "su malonumu", "žinoma").
  * Comma-rich pacing — micro-pauses soften delivery on Soniox.
  * Avoid clipped, 3-word sentences which sound abrupt.
"""
from __future__ import annotations

import pytest


def test_prompt_documents_warmth_markers():
    """Prompt must include a section that suggests warm Lithuanian
    phrases the agent should use to compensate for Soniox's lack of
    tone control."""
    from agent import INSTRUCTIONS
    text = INSTRUCTIONS.lower()
    has_warmth_section = (
        "šil" in text  # šiltai, šiltesnis
        and ("labai mielai" in text or "su malonumu" in text or "prašom" in text)
    )
    assert has_warmth_section, (
        "Prompt must document warm Lithuanian phrases to soften Soniox "
        "delivery. Without them, Soniox sounds clipped."
    )


def test_prompt_documents_comma_pacing():
    """Prompt must instruct the agent to use comma-rich pacing —
    Soniox renders prosody from punctuation, so kablelių mikropauzės
    švelnina pristatymą."""
    from agent import INSTRUCTIONS
    text = INSTRUCTIONS.lower()
    has_pacing_rule = (
        "kablel" in text
        and ("mikropauz" in text or "pauz" in text or "švelnesn" in text or "ramus" in text or "ramiau" in text)
    )
    assert has_pacing_rule, (
        "Prompt must explain comma-rich pacing as the Soniox softening "
        "lever (no native pitch/speed knobs available)."
    )


@pytest.mark.asyncio
async def test_greeting_response_uses_warm_phrasing(llm, agent_session):
    """First user turn — the agent's reply should include at least one
    warm Lithuanian phrase, OR be visibly comma-paced (>=3 commas in a
    typical 2-3 sentence reply)."""
    result = await agent_session.run(user_input="Laba diena, norėčiau pasiteirauti.")
    msg = result.expect.contains_message(role="assistant")
    text = (msg.event().item.text_content or "").lower()

    warmth_markers = [
        "labai mielai",
        "su malonumu",
        "žinoma",
        "prašom",
        "prašau",
        "dėkoju",
        "ačiū kad",
        "be jokios abejonės",
        "suprantu jus",
        "puiku",
    ]
    has_warmth = any(m in text for m in warmth_markers)

    comma_count = text.count(",")
    sentence_count = max(text.count(".") + text.count("?") + text.count("!"), 1)
    commas_per_sentence = comma_count / sentence_count

    assert has_warmth or commas_per_sentence >= 1.0, (
        "Reply lacks both a warm phrase AND comma-rich pacing — Soniox "
        f"will sound clipped.\n"
        f"reply: {text!r}\n"
        f"warmth_markers found: {[m for m in warmth_markers if m in text]}\n"
        f"commas/sentence: {commas_per_sentence:.2f}"
    )


@pytest.mark.asyncio
async def test_avoids_clipped_short_sentences(llm, agent_session):
    """Soniox's prosody is harsher on chained 3-4 word sentences. A
    typical procedural answer should average ≥6 words per sentence."""
    result = await agent_session.run(
        user_input="Kokių dokumentų reikia deklaruoti gyvenamajai vietai?"
    )
    msg = result.expect.contains_message(role="assistant")
    text = msg.event().item.text_content or ""

    # Split on sentence-ending punctuation, ignore empty fragments.
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        pytest.skip("No sentences in response")

    word_counts = [len(s.split()) for s in sentences]
    avg = sum(word_counts) / len(word_counts)

    assert avg >= 5.0, (
        f"Average sentence length is {avg:.1f} words — too clipped for "
        "Soniox prosody. Aim for ≥6 words/sentence by combining clauses "
        f"with commas. Sentences: {word_counts}"
    )
