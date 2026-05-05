"""Vilnius savivaldybės balso agentas — Soniox STT + Soniox TTS variantas.

Vienas failas: AgentSession setup, sistemos prompt'as, balso įrankiai
(KB lookup'ai), entrypoint.

- STT: Soniox ``stt-rt-v4`` (lt strict).
- LLM: ``openai/gpt-5.3-chat-latest`` per LiveKit Inference.
- TTS: Soniox ``tts-rt-v1-preview`` per lokalų ``sioniox/`` paketą
  (žr. soniox_livekit_agent projektą — tas pats plugin'as).
- Turn detection: Soniox STT endpointing (``turn_detection="stt"``).
  JOKIO turn-detector modelio.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    inference,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import RunContext
from livekit.plugins import silero, soniox

import sioniox

from knowledge.faqs import FAQS, faq_index, get_faq
from knowledge.kb import (
    CONTACTS,
    DEADLINES,
    KONSULTACIJOS,
    NELAIKOMI_PAKEITUSIAIS,
    REQUIRED_DOCS,
    SENIUNIJOS_INFO,
    SENIUNIJOS_VILNIUJE,
    SERVICE_OVERVIEW,
)


load_dotenv(dotenv_path=".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vilnius-agent")


# ---------------------------------------------------------------------------
# Call state
# ---------------------------------------------------------------------------

@dataclass
class CallState:
    last_topic: Optional[str] = None
    questions_answered: int = 0


RunCtx = RunContext[CallState]


# ---------------------------------------------------------------------------
# Soniox STT context — lietuvių municipalinės terminologijos prior'as
# ---------------------------------------------------------------------------

_LT_DIGITS = ["nulis", "vienas", "du", "trys", "keturi", "penki", "šeši", "septyni", "aštuoni", "devyni"]

STT_CONTEXT = {
    "text": (
        "Tai skambutis į Vilniaus miesto savivaldybės informacijos liniją. "
        "Skambinantis klausia apie gyvenamosios vietos deklaravimą, reikiamus "
        "dokumentus, terminus, seniūnijas, vaiko deklaravimą, išvykimą iš "
        "Lietuvos Respublikos, nuomos ir panaudos sutartis, savininko sutikimus."
    ),
    "terms": [
        "Vilnius", "Vilniaus miesto savivaldybė", "savivaldybė", "deklaravimas",
        "gyvenamoji vieta", "seniūnija", "Klientų aptarnavimo skyrius",
        "asmens tapatybės kortelė", "pasas", "gimimo liudijimas", "asmens kodas",
        "nuomos sutartis", "panaudos sutartis", "savininko sutikimas",
        "globėjas", "rūpintojas", "įtėvis",
        "epaslaugos", "Elektroniniai valdžios vartai",
        *_LT_DIGITS,
        *[f"{name} seniūnija" for name in SENIUNIJOS_VILNIUJE],
    ],
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

INSTRUCTIONS = f"""Tu esi Rūta — Vilniaus miesto savivaldybės virtuali balso asistentė, atsakanti TIK į klausimus apie GYVENAMOSIOS VIETOS DEKLARAVIMĄ. Kalbi lietuviškai, šiltai, mandagiai, trumpai (1–3 sakiniai).

# Pokalbio scenarijus

1. Klientas užduoda klausimą.
2. Identifikuoji temą ir kvieti VIENĄ įrankį, kad gautum kanoninį tekstą:
   - ``lookup_faq(faq_id)`` — DUK atsakymas pagal slug'ą iš sąrašo žemiau.
   - ``get_required_documents(scenario)`` — dokumentų sąrašas.
   - ``get_deadline(scenario)`` — terminas.
   - ``get_contact_info()`` — telefonai, adresai, darbo laikas.
   - ``get_seniunijos_info()`` — kur deklaruoti pagal teritoriją (Šnipiškių išimtis).
   - ``get_service_overview()`` — bendras paslaugos aprašymas.
   - ``get_exempt_categories()`` — kas nelaikomas pakeitusiu gyvenamąją vietą.
3. Pristatyk atsakymą SAVO ŽODŽIAIS, trumpai, šiltai.
4. Klausi „ar dar galiu kuo nors padėti?".

# Apribojimai

Atsakai TIK apie gyvenamosios vietos deklaravimą Vilniuje. Jei klientas
klausia apie pasus, mokesčius, darželius — mandagiai pasakyk, kad ši
tema už tavo kompetencijos ribų, nukreipk į bendrą savivaldybės numerį
(penki, du vienas vienas, du tūkstančiai) arba svetainę vilnius taškas
l-t, ir paklausk, ar gali padėti su deklaravimu.

# TTS taisyklės

Tavo tekstą skaitys Soniox TTS lietuvių kalba.

- Skaičius ir santrumpas perskaityk PILNAIS lietuviškais žodžiais:
  „septynios darbo dienos" (ne „7 d.d."), „elektroniniu paštu" (ne
  „el. paštu"), „pavyzdžiui" (ne „pvz."), „rajonas" (ne „r.").
- URL adresus sakyk skiemenimis: „e paslaugos taškas l t" (ne
  „www.epaslaugos.lt").
- Telefono numerius — žodžiais grupėmis: „penki, du vienas vienas, du
  tūkstančiai".
- Niekada nevartok simbolių „/", „—", skliaustų „(", „)", „+".
- Atsakymai trumpi (1–3 sakiniai), be markdown'o, be emoji, BE
  laužtinių skliaustų žymų — Soniox TTS jų neinterpretuoja.

# Bendros taisyklės

- NIEKADA neišgalvok dokumentų, terminų ar kontaktų — visada gauk juos iš įrankio.
- Šnipiškių gyventojai deklaruoja Klientų aptarnavimo skyriuje, NE seniūnijoje.
- Deklaravimo paslauga yra NEMOKAMA.
- Jei klientas aiškiai prašo žmogaus — pasiūlyk paskambinti bendruoju
  savivaldybės numeriu penki, du vienas vienas, du tūkstančiai.

# DUK indeksas (``lookup_faq`` slug'ai)

{faq_index()}
"""


GREETING = (
    "Sveiki, čia Vilniaus miesto savivaldybės virtuali asistentė Rūta. "
    "Padėsiu dėl gyvenamosios vietos deklaravimo. Klausau."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class InfoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        await self.session.say(GREETING, allow_interruptions=False)

    @function_tool
    async def lookup_faq(self, context: RunCtx, faq_id: str) -> str:
        """Grąžina kanoninį atsakymą iš Vilniaus savivaldybės DUK pagal slug'ą."""
        faq = get_faq(faq_id)
        if faq is None:
            available = ", ".join(FAQS.keys())
            return f"Slug'as {faq_id!r} nerastas. Galimi: {available}."
        context.userdata.last_topic = faq_id
        context.userdata.questions_answered += 1
        return (
            f"DUK tema: {faq['question']}\n"
            f"Kanoninis atsakymas: {faq['answer']}\n\n"
            "INSTRUKCIJA TAU: pristatyk savo žodžiais, trumpai, šiltai."
        )

    @function_tool
    async def get_required_documents(self, context: RunCtx, scenario: str) -> str:
        """Reikiamų dokumentų sąrašas pagal scenarijų.

        Galimi: ``persikraustymas_lt``, ``atvykimas_i_lt``, ``isvykimas_iz_lt``,
        ``vaiko_deklaravimas``, ``globejas``, ``studentas_bendrabutyje``.
        """
        docs = REQUIRED_DOCS.get(scenario)
        if docs is None:
            return f"Scenarijus {scenario!r} nepalaikomas. Galimi: {', '.join(REQUIRED_DOCS)}."
        context.userdata.last_topic = f"docs:{scenario}"
        context.userdata.questions_answered += 1
        numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(docs, 1))
        return f"Reikiami dokumentai:\n{numbered}"

    @function_tool
    async def get_deadline(self, context: RunCtx, scenario: str) -> str:
        """Deklaravimo terminas pagal scenarijų.

        Galimi: ``persikraustymas_lt``, ``atvykimas_i_lt``, ``isvykimas_iz_lt``,
        ``be_pakeitimo``.
        """
        deadline = DEADLINES.get(scenario)
        if deadline is None:
            return f"Scenarijus {scenario!r} nežinomas. Galimi: {', '.join(DEADLINES)}."
        context.userdata.last_topic = f"deadline:{scenario}"
        context.userdata.questions_answered += 1
        return deadline

    @function_tool
    async def get_contact_info(self, context: RunCtx) -> str:
        """Vilniaus savivaldybės kontaktai deklaravimo klausimais."""
        context.userdata.last_topic = "contact_info"
        context.userdata.questions_answered += 1
        return (
            f"Bendras savivaldybės telefonas: {CONTACTS['savivaldybes_telefonas']} "
            f"(trumpasis {CONTACTS['savivaldybes_trumpasis']}). "
            f"Elektroninis paštas: {CONTACTS['savivaldybes_elpastas']}. "
            f"Adresas: {CONTACTS['savivaldybes_adresas']}. "
            f"Konsultacijos telefonu: {CONTACTS['konsultacijos_telefonas']}, "
            f"{CONTACTS['konsultacijos_darbo_laikas']}. "
            f"Elektroninės paslaugos: {CONTACTS['epaslaugos_url']}. "
            f"Seniūnijos: {CONTACTS['seniunijos_url']}.\n\n"
            "INSTRUKCIJA TAU: pristatyk klientui TIK tuos kontaktus, "
            "kurių jis prašo. Numerius sakyk lietuviškais žodžiais grupėmis."
        )

    @function_tool
    async def get_seniunijos_info(self, context: RunCtx) -> str:
        """Kur deklaruoti pagal Vilniaus teritorijos dalį (Šnipiškių išimtis)."""
        context.userdata.last_topic = "seniunijos"
        context.userdata.questions_answered += 1
        return SENIUNIJOS_INFO

    @function_tool
    async def get_service_overview(self, context: RunCtx) -> str:
        """Bendras gyvenamosios vietos deklaravimo paslaugos aprašymas."""
        context.userdata.last_topic = "overview"
        context.userdata.questions_answered += 1
        return f"{SERVICE_OVERVIEW}\n\n{KONSULTACIJOS}"

    @function_tool
    async def get_exempt_categories(self, context: RunCtx) -> str:
        """Asmenys, NELAIKOMI pakeitusiais deklaruotą gyvenamąją vietą.

        Pavyzdžiui: studentai mokymosi laikotarpiu, jūreiviai, asmenys
        atliekantys karinę tarnybą, gydomi stacionariose įstaigose.
        """
        context.userdata.last_topic = "exempt"
        context.userdata.questions_answered += 1
        numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(NELAIKOMI_PAKEITUSIAIS, 1))
        return f"Šie asmenys nelaikomi pakeitusiais gyvenamąją vietą:\n{numbered}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"Starting Vilnius (Soniox TTS) agent in room {ctx.room.name}")
    await ctx.connect()

    session = AgentSession[CallState](
        userdata=CallState(),
        vad=silero.VAD.load(),
        stt=soniox.STT(
            params=soniox.STTOptions(
                model="stt-rt-v4",
                language_hints=["lt"],
                language_hints_strict=True,
                context=STT_CONTEXT,
                max_endpoint_delay_ms=1500,
            ),
        ),
        llm=inference.LLM(model="openai/gpt-5.3-chat-latest"),
        # Soniox TTS — local sioniox/ package (custom plugin, ported from
        # the soniox_livekit_agent reference project). Lithuanian voice
        # "Adrian" via Soniox's tts-rt-v1-preview model.
        tts=sioniox.TTS(
            language="lt",
            voice="Adrian",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            interruption={"resume_false_interruption": True},
            preemptive_generation={"enabled": True},
        ),
        max_tool_steps=3,
        user_away_timeout=30,
    )

    await session.start(agent=InfoAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="vilnius-soniox"))
