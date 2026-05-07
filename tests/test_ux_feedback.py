"""UX-feedback regression tests — locked-in behaviors based on real
production traces (RM_y3hinG3djZUB, RM_AYNAmprLgMzu, May 2026).

User feedback verbatim:
    "agent is not polite and repeats itself, does not provide answers
     proactively, answers are limited."

Each test below is one specific, observable failure mode from those
traces. They MUST pass before a new build is shipped.
"""
from __future__ import annotations

import re
import pytest


# ---------------------------------------------------------------------------
# Static (non-LLM) lints on the agent's instructions
# ---------------------------------------------------------------------------

def test_prompt_has_no_default_followup_phrase():
    """The system prompt must NOT mandate ending every reply with
    'Ar dar galiu kuo nors padėti?'. In the production trace this
    follow-up appeared in 100% of assistant turns and felt dismissive.

    The new contract: agent uses the follow-up sparingly — only when the
    user finished a topic AND hasn't signalled a goodbye.
    """
    from agent import INSTRUCTIONS

    # Allow the phrase to exist as guidance, but it must not be coupled
    # with "always" / "every reply" / "kiekvieną" wording.
    bad_pairs = [
        ("kiekvien", "ar dar galiu"),
        ("visada", "ar dar galiu"),
        ("pabaigoje", "ar dar galiu"),
    ]
    text = INSTRUCTIONS.lower()
    bad_found = [
        f"'{a}' near 'ar dar galiu'"
        for a, b in bad_pairs
        if a in text and b in text and abs(text.find(a) - text.find(b)) < 80
    ]
    assert not bad_found, (
        "Prompt couples 'always' with 'Ar dar galiu kuo nors padėti?' — "
        f"this caused the dismissive-loop UX issue. Found: {bad_found}"
    )


def test_prompt_documents_end_of_call_behavior():
    """The prompt must explicitly tell the agent how to handle goodbye
    signals from the user (Ačiū / Iki / Gerai, ate / Gražios dienos /
    padėkim ragelį) — namely, say a SHORT farewell and STOP repeating
    the offer to help."""
    from agent import INSTRUCTIONS

    text = INSTRUCTIONS.lower()
    # At least one of these closing-protocol cues must be present.
    has_closing_rule = any(
        cue in text
        for cue in [
            "atsisveikin",
            "kai klientas atsisveikina",
            "padėkim ragel",
            "užbaig",
            "iki / ate / ačiū",
            "ačiū / iki / ate",
        ]
    )
    assert has_closing_rule, (
        "Prompt does not document end-of-call behavior — agent loops "
        "'kreipkitės' farewells indefinitely (see RM_AYNAmprLgMzu)."
    )


def test_prompt_documents_proactive_clarification():
    """When asked vague 'kaip galite padėti?', the agent must
    proactively narrow scope (persikraustote / atvykstate / išvykstate)
    instead of reciting a flat list of capabilities."""
    from agent import INSTRUCTIONS

    text = INSTRUCTIONS.lower()
    has_proactive_rule = (
        "proaktyv" in text
        or "patikslin" in text
        or "atskleidžiantis klausim" in text
        or ("scenarij" in text and "klausk" in text)
    )
    assert has_proactive_rule, (
        "Prompt does not require proactive clarification — agent gives "
        "generic answers when it should narrow scope (see real trace "
        "RM_AYNAmprLgMzu, turn 'Padėk. Kaip gali padėti?')."
    )


# ---------------------------------------------------------------------------
# LLM-judged behavioral tests — drive real conversations and grade them
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_does_not_loop_followup_when_user_says_goodbye(llm, agent_session):
    """User says 'Ačiū' then 'Gražios dienos'. Agent must say a short
    farewell and NOT keep offering more help on subsequent turns."""
    # Turn 1: a normal info question to set up context
    await agent_session.run(user_input="Sveiki, ką galiu deklaruoti internetu?")
    # Turn 2: user begins to close
    await agent_session.run(user_input="Ačiū už pagalbą.")
    # Turn 3: user finalizes
    result = await agent_session.run(user_input="Gražios dienos.")

    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas aiškiai užbaigia pokalbį. Asistentės atsakymas "
            "turi būti TRUMPAS atsisveikinimas (pavyzdžiui „Gražios "
            "dienos!“ arba „Iki, sėkmės deklaruojant“). NETURI būti "
            "klausimo „Ar dar galiu kuo nors padėti?“ ar pasiūlymo "
            "kreiptis vėliau, nes klientas jau pasakė atsisveikinimo "
            "frazę."
        ),
    )


@pytest.mark.asyncio
async def test_proactively_clarifies_on_vague_help_request(llm, agent_session):
    """User asks vague 'Kaip galite padėti?'. Agent should ask back to
    narrow the scenario (persikraustote / atvykstate / išvykstate)
    instead of dumping a list."""
    result = await agent_session.run(user_input="Kaip galite man padėti?")
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas užduoda BENDRĄ klausimą — asistentė turi "
            "PROAKTYVIAI patikslinti situaciją: paklausti, ar klientas "
            "persikrausto Lietuvos viduje, atvyko į Lietuvą, išvyksta, "
            "deklaruoja vaiką, ar yra kita situacija. "
            "Atsakymas TURI BŪTI klausimas (arba kelios alternatyvos), "
            "o ne tiesiog sąrašas „galiu paaiškinti, kaip…“."
        ),
    )


@pytest.mark.asyncio
async def test_does_not_repeat_facts_within_conversation(llm, agent_session):
    """Real trace: user asked 'Koks internetinis adresas, sakei?' and
    agent re-stated the same URL twice in two consecutive turns. After
    the first answer, follow-ups should NOT re-state the same fact —
    they should add NEW info or move on."""
    # Turn 1: agent gives the URL
    r1 = await agent_session.run(user_input="Kaip deklaruoti internetu?")
    msg1 = r1.expect.contains_message(role="assistant")
    text1 = (msg1.event().item.text_content or "").lower()
    assert "epaslaugos" in text1 or "e paslaugos" in text1

    # Turn 2: user just acknowledges. Per the new prompt, the agent is
    # ALLOWED to stay silent here (no assistant message at all). If the
    # agent does reply, the reply must NOT re-state the same URL.
    r2 = await agent_session.run(user_input="Aha, supratau.")
    assistant_events = [
        e for e in r2.events
        if hasattr(e, "item") and getattr(e.item, "role", None) == "assistant"
    ]
    if not assistant_events:
        # Silence is fine — that's exactly what the prompt prescribes.
        return
    text2 = (assistant_events[0].item.text_content or "").lower()
    assert "epaslaugos" not in text2 and "e paslaugos" not in text2, (
        f"Agent re-stated URL on a mere acknowledgement turn: {text2!r}"
    )


@pytest.mark.asyncio
async def test_substantive_answer_for_typical_question(llm, agent_session):
    """User's complaint: 'answers are limited'. For a realistic question
    like 'Kokius dokumentus reikia kai persikraustau?' the agent must
    give 2–4 concrete documents, not a one-liner."""
    result = await agent_session.run(
        user_input="Kokių dokumentų man reikės, jeigu ką tik persikrausčiau į naują butą Vilniuje?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas klausia konkrečių dokumentų persikraustant į "
            "naują butą Vilniuje. Asistentės atsakymas TURI būti "
            "SUBSTANCIALUS — paminėti BENT DVI šias dokumentų grupes: "
            "asmens tapatybės dokumentą (pasą arba asmens tapatybės "
            "kortelę); IR teisės gyventi būste pagrindą (pvz., nuomos "
            "ar panaudos sutartis arba savininko sutikimas, ar "
            "nuosavybės dokumentas). Atsakymas TURI būti naudingas "
            "klientui, ne paviršutiniškas — bent kelios konkrečios "
            "detalės. Atsakymo ilgis nuo dviejų sakinių iki kelių "
            "trumpų eilučių sąrašo. Bendrybė „reikės asmens dokumento“ "
            "viena pati nepakanka."
        ),
    )


@pytest.mark.asyncio
async def test_handles_human_handoff_request_gracefully(llm, agent_session):
    """Real trace: user asked 'Ar galite sujungti su savivaldybės ne
    virtualiu konsultantu?'. Agent must apologize, give the live phone
    number, and acknowledge it can't transfer — without abruptly
    pushing 'Ar dar galiu padėti?'."""
    result = await agent_session.run(
        user_input="Norėčiau pakalbėti su gyvu savivaldybės darbuotoju, ne su robotu."
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas prašo perduoti gyvam darbuotojui. Asistentė turi: "
            "1) ramiai paaiškinti, kad ji virtuali ir negali perjungti "
            "skambučio; "
            "2) nurodyti BENDRĄ savivaldybės telefoną — penki, du "
            "vienas vienas, du tūkstančiai, ARBA konsultacijų telefoną "
            "penki, du penki devyni penki penki aštuoni vienas; "
            "3) atsakymas turi būti EMPATINIS, ne mechaninis. "
            "Klausimo „Ar dar galiu kuo nors padėti?“ tame atsakyme "
            "BŪTI NETURI — tai jau atsisveikinimo zona."
        ),
    )


@pytest.mark.asyncio
async def test_does_not_emit_combining_accent_chars(llm, agent_session):
    """Real trace: agent rendered 'Septýni' / 'Rū́ta' with
    U+0301 COMBINING ACUTE ACCENT. Soniox TTS can't pronounce those —
    they leak into audio as glitches. Agent must NEVER emit them."""
    forbidden = [
        "́",  # combining acute
        "̀",  # combining grave
        "̃",  # combining tilde
        "̈",  # combining diaeresis
    ]

    # Force a context where past traces showed the agent reaching for
    # accent marks (numbers / pronouncing names)
    inputs = [
        "Pasakyk skaičių septyni.",
        "Koks tavo vardas?",
        "Ištark mano pavardę: Petraitis.",
    ]
    for q in inputs:
        result = await agent_session.run(user_input=q)
        msg = result.expect.contains_message(role="assistant")
        text = msg.event().item.text_content or ""
        leaked = [hex(ord(c)) for c in text if c in forbidden]
        assert not leaked, (
            f"Agent emitted combining accent chars {leaked} in reply "
            f"to {q!r}. Soniox cannot render these. Reply: {text!r}"
        )


@pytest.mark.asyncio
async def test_polite_when_user_complains(llm, agent_session):
    """Real trace: user said 'Labai nemandagiai mane aptarnaujate' and
    agent's apology was thin. The recovery must be warm and acknowledge
    the feedback explicitly, not deflect."""
    # Set up a normal turn first
    await agent_session.run(user_input="Kaip deklaruotis internetu?")
    # Then the complaint
    result = await agent_session.run(
        user_input="Sakei, sakei... atsakymai labai nemandagūs ir riboti."
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Klientas išreiškė nepasitenkinimą („nemandagūs ir riboti“). "
            "Asistentės atsakymas TURI: "
            "1) PRIIMTI grįžtamąjį ryšį empatiškai — pavyzdžiui, "
            "„Suprantu jus“, „Pabandysiu išsamiau“. "
            "Sausa frazė „atsiprašau, jei taip pasirodė“ NĖRA reikalinga, "
            "bet jei yra — irgi tinka. "
            "2) Pateikti IŠSAMESNĮ atsakymą į ankstesnį klausimą "
            "(daugiau detalių, konkrečių žingsnių) ARBA pasiūlyti gyvo "
            "konsultanto telefoną. "
            "NEPRIIMTINA: trumpa standartinė frazė + iškart „ar dar "
            "galiu kuo nors padėti?“ be jokio papildomo turinio."
        ),
    )
