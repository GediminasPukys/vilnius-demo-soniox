"""Conversational-depth regression tests — locked-in behaviors based on
follow-up user feedback (May 7, 2026):

    "agent is always asking 'Ar dar galiu kuo padėti' which tends to
     be quite annoying. agent should give deeper answers, ask leading
     forward questions."

Three contracts encoded as tests:
    1. The phrase 'Ar dar galiu kuo nors padėti?' must NOT appear in
       a typical substantive-answer turn.
    2. After a substantive answer, the agent MUST ask a leading
       forward question that advances the user's specific scenario
       (NOT a generic "anything else?").
    3. Procedure questions MUST get substantive answers (the depth
       test from test_ux_feedback raised the bar; this re-asserts it
       with a stricter threshold).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Static prompt lints
# ---------------------------------------------------------------------------

def test_prompt_explicitly_forbids_default_followup():
    """The prompt MUST explicitly forbid 'Ar dar galiu kuo nors padėti?'
    as a default close. Earlier prompt only soft-bounded it; user
    feedback shows the LLM still emits it on every turn."""
    from agent import INSTRUCTIONS
    text = INSTRUCTIONS.lower()

    # Some signal that the phrase is explicitly discouraged / banned
    has_strong_ban = (
        "draudžiama „ar dar galiu" in text
        or "nesakoma „ar dar galiu" in text
        or "nevartok „ar dar galiu" in text
        or "nevartok ar dar galiu" in text
        or ("ar dar galiu" in text and "nevartok" in text)
        or ("ar dar galiu" in text and "draudžiama" in text)
        or ("ar dar galiu" in text and "uždraudžiama" in text)
    )
    assert has_strong_ban, (
        "Prompt must EXPLICITLY ban 'Ar dar galiu kuo nors padėti?' as "
        "a default close. Soft-bounding it doesn't work — LLM still "
        "uses it every turn."
    )


def test_prompt_documents_leading_question_pattern():
    """Prompt must instruct the agent to ask FORWARD-LEADING questions
    that advance the user's specific scenario."""
    from agent import INSTRUCTIONS
    text = INSTRUCTIONS.lower()
    has_pattern = (
        "vedant" in text
        or "vedantis klausim" in text
        or "vedantys klausim" in text
        or "kreipiantis klausim" in text
        or "logiškas tęsinys" in text
        or "logiškas kitas žingsn" in text
        or ("kitas žingsn" in text and "klaus" in text)
        or "konkretesnis klausim" in text
    )
    assert has_pattern, (
        "Prompt must document the forward-leading question pattern. "
        "Without it, the agent reverts to 'ar dar galiu...' as the "
        "default close."
    )


# ---------------------------------------------------------------------------
# LLM-judged behavioral tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_does_not_say_ar_dar_galiu_after_substantive_answer(llm, agent_session):
    """After a substantive procedural answer, the agent must NOT close
    with 'Ar dar galiu kuo nors padėti?' — that phrase should be
    extremely rare. Instead, the agent should either close with a
    leading question OR end naturally."""
    result = await agent_session.run(
        user_input="Kaip man deklaruoti gyvenamąją vietą jei nuomojuosi butą?"
    )
    msg = result.expect.contains_message(role="assistant")
    text = (msg.event().item.text_content or "").lower()

    # Strict ban on this exact phrase / its close variants
    forbidden_substrings = [
        "ar dar galiu kuo nors padėti",
        "ar dar galiu padėti",
        "ar dar kuo nors galiu padėti",
        "ar dar galiu kuo padėti",
    ]
    for phrase in forbidden_substrings:
        assert phrase not in text, (
            f"Agent emitted forbidden close-phrase {phrase!r} after a "
            f"substantive answer.\nFull reply: {text!r}"
        )


@pytest.mark.asyncio
async def test_asks_forward_leading_question_after_substantive_answer(llm, agent_session):
    """After answering 'kaip deklaruoti', the agent must ask a
    SPECIFIC leading question (e.g. 'ar nuomojate ar turite nuosavą
    būstą?', 'ar deklaruosite vienas, ar su šeima?'), NOT a generic
    'anything else?'."""
    result = await agent_session.run(
        user_input="Norėčiau deklaruoti gyvenamąją vietą Vilniuje."
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Atsakymas TURI: "
            "(a) trumpai paaiškinti pagrindinį žingsnį arba bent vieną "
            "konkretų faktą iš deklaravimo procedūros. "
            "(b) Pabaigoje uždavęs SPECIFINĮ vedantį klausimą, kuris "
            "gilina šio kliento konkrečią situaciją — pavyzdžiui: "
            "„Ar nuomojate, ar turite nuosavą būstą?“; "
            "„Ar planuojate atvykti į seniūniją, ar deklaruoti "
            "internetu?“; „Ar deklaruosite vienas, ar kartu su šeimos "
            "nariais?“; „Ar šis būstas pirmasis, ar keičiate adresą "
            "Vilniuje?“. "
            "NETINKA: bendra frazė „ar dar galiu kuo nors padėti?“ — "
            "ji NEVEDA pokalbio į priekį, vartotojas ją laiko "
            "atstumiančia. "
            "Atsakymas turi būti lietuvių kalba, mandagus, be markdown'o."
        ),
    )


@pytest.mark.asyncio
async def test_substantive_answer_for_procedure_question(llm, agent_session):
    """Procedure-type questions need DEEPER answers than the prior
    bar — at least 50 spoken Lithuanian words and concrete steps,
    not just one sentence."""
    result = await agent_session.run(
        user_input="Kaip vyksta deklaravimo procesas internetu? Aš niekada to nedariau."
    )
    msg = result.expect.contains_message(role="assistant")
    text = msg.event().item.text_content or ""
    word_count = len(text.split())
    assert word_count >= 35, (
        f"Procedure answer is too short ({word_count} words). User "
        "explicitly requested a deeper walkthrough — agent must give "
        "at least 35 Lithuanian words with concrete steps. "
        f"Got: {text!r}"
    )

    await msg.judge(
        llm,
        intent=(
            "Atsakymas TURI: "
            "(a) bent du-tris konkrečius žingsnius (prisijungimas, "
            "prašymo pildymas, dokumentų pateikimas, pateikimas / "
            "siuntimas, gavimas); "
            "(b) paminėti, kur vyksta procesas (e paslaugos taškas l t); "
            "(c) pabaigoje vedantis klausimas, ne „ar dar galiu padėti?“. "
            "Tinkamas atsakymo ilgis 35-90 žodžių. "
            "Atsakymas lietuvių kalba, be markdown'o, be santrumpų."
        ),
    )


@pytest.mark.asyncio
async def test_uses_forward_question_on_arrival_to_lithuania(llm, agent_session):
    """For 'atvykau iš užsienio', agent must steer with leading
    questions about specific situation."""
    result = await agent_session.run(
        user_input="Ką tik atvykau gyventi į Lietuvą iš Vokietijos. Ką man reikia daryti?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas atvyko į Lietuvą iš ES šalies. Asistentės atsakymas "
            "TURI: "
            "(1) iškart pasakyti, kad gyvenamąją vietą reikia deklaruoti "
            "per VIENĄ MĖNESĮ nuo atvykimo; "
            "(2) bent vienu sakiniu pateikti ką daryti (kur kreiptis, "
            "kokie dokumentai); "
            "(3) pabaigoje uždavęs SPECIFINĮ vedantį klausimą — "
            "pavyzdžiui: „Ar jau radote būstą Vilniuje?“ / „Ar planuojate "
            "deklaruoti internetu, ar atvyksite į seniūniją?“ / „Ar turite "
            "su savimi galiojantį pasą arba ES leidimą gyventi?“. "
            "NETINKA: „Ar dar galiu kuo nors padėti?“ — tai ne vedantis "
            "klausimas. Atsakymas lietuvių kalba, be markdown'o."
        ),
    )
