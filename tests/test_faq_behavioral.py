"""Behavioral tests — 15 FAQ Q&A pairs are the canonical test corpus.

Each FAQ becomes one parametrised test that drives the natural-language
question into ``InfoAgent`` and grades the spoken reply with ``judge(llm)``.
"""
from __future__ import annotations

import pytest

from knowledge.faqs import FAQS


@pytest.mark.parametrize("faq_id", list(FAQS.keys()))
@pytest.mark.asyncio
async def test_info_agent_answers_faq(faq_id, llm, agent_session):
    faq = FAQS[faq_id]
    result = await agent_session.run(user_input=faq["question"])
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            f"Atsako į klausimą: {faq['question']}. "
            "Atsakymo pagrindinis teiginys turi būti suderinamas su šiuo "
            "kanoniniu tekstu (atsakymas gali būti kur kas trumpesnis "
            f"ir savais žodžiais — taip ir TURI būti): {faq['answer']}. "
            "Atsakymas turi būti lietuvių kalba, mandagus, trumpas, "
            "be markdown'o, be emoji, be santrumpų tipo el., d.d., r. "
            "Atsakymas NETURI PRIEŠTARAUTI kanoniniam tekstui. "
            "Smulkių detalių praleidimas YRA leistinas. "
            "Papildomos teisingos pagalbos (pavyzdžiui, paminėti, kad "
            "reikia teisėto pagrindo gyventi būste) yra LEISTINOS. "
            "Jei minimas URL adresas, tinka tiek balsui pritaikyta forma "
            "(pavyzdžiui, „e paslaugos taškas l t“), tiek bendresnis "
            "aprašymas (pavyzdžiui, „valstybės paslaugų portalas“)."
        ),
    )


@pytest.mark.asyncio
async def test_agent_cites_seven_working_days_when_leaving_lt(llm, agent_session):
    result = await agent_session.run(
        user_input="Išvykstu į užsienį maždaug metams — ką turiu padaryti?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Atsakymas turi pasakyti, kad reikia deklaruoti išvykimą iš "
            "Lietuvos PER SEPTYNIAS DARBO DIENAS IKI IŠVYKIMO. Skaičius "
            "pasakomas LIETUVIŠKAIS ŽODŽIAIS, ne santrumpa."
        ),
    )


@pytest.mark.asyncio
async def test_agent_cites_one_month_when_arriving(llm, agent_session):
    result = await agent_session.run(
        user_input="Ką tik persikrausčiau į naują butą Vilniuje. Per kiek laiko turiu deklaruoti?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Atsakymas turi paminėti VIENĄ MĖNESĮ kaip terminą gyvenamosios "
            "vietos deklaravimui. Pradžios momentas gali būti formuluojamas "
            "kaip nuo persikraustymo, nuo apsigyvenimo arba nuo pareigos "
            "atsiradimo — visi yra teisingi. Atsakymas lietuvių kalba, "
            "žodžiais, be santrumpų."
        ),
    )


@pytest.mark.asyncio
async def test_agent_redirects_when_off_topic(llm, agent_session):
    result = await agent_session.run(
        user_input="Sakykite, kaip man pasidaryti naują pasą?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Atsakymas mandagiai pasako, kad ši tema yra už asistentės "
            "kompetencijos ribų / kad ji padeda tik dėl gyvenamosios "
            "vietos deklaravimo, ir nukreipia klientą kitur. NETURI duoti "
            "konkrečios informacijos apie tai, kaip pasidaryti pasą."
        ),
    )


@pytest.mark.asyncio
async def test_agent_handles_snipiskes_special_case(llm, agent_session):
    result = await agent_session.run(
        user_input="Aš gyvenu Šnipiškėse. Į kurią seniūniją man kreiptis dėl deklaravimo?"
    )
    msg = result.expect.contains_message(role="assistant")
    await msg.judge(
        llm,
        intent=(
            "Atsakymas turi paaiškinti, kad Šnipiškių gyventojai "
            "deklaruojasi NE seniūnijoje, o Vilniaus miesto savivaldybės "
            "Klientų aptarnavimo skyriuje."
        ),
    )
