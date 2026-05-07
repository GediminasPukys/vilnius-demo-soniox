"""Vilnius savivaldybės balso agentas — Soniox STT + Soniox TTS variantas.

Vienas failas: AgentSession setup, sistemos prompt'as, entrypoint.

KB FILOSOFIJA: visa žinių bazė (FAQ + dokumentai + terminai + kontaktai)
INLINE'ina į sistemos prompt'ą — JOKIŲ funkcijų įrankių. Su LiveKit
Inference prompt-caching'u 2 500-token'ų KB blokas po pirmo skambučio
yra cache'uojamas, todėl kaina lieka stabili, o latency'is sumažėja
nuo dviejų LLM round-trip'ų į vieną.

- STT: Soniox ``stt-rt-v4`` (lt strict).
- LLM: ``openai/gpt-5.3-chat-latest`` per LiveKit Inference.
- TTS: Soniox ``tts-rt-v1`` per lokalų ``soniox_tts/`` paketą.
- Turn detection: Soniox STT endpointing (``turn_detection="stt"``).
  JOKIO turn-detector modelio.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import AsyncIterable, Optional

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
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.voice import ModelSettings, RunContext, room_io
from livekit.plugins import noise_cancellation, silero, soniox

import soniox_tts

from knowledge.kb import KB_TEXT, SENIUNIJOS_VILNIUJE


load_dotenv(dotenv_path=".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("vilnius-agent")


# ---------------------------------------------------------------------------
# Call state
# ---------------------------------------------------------------------------

@dataclass
class CallState:
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

INSTRUCTIONS = f"""Tu esi Rūta — Vilniaus miesto savivaldybės virtuali balso asistentė, atsakanti TIK į klausimus apie GYVENAMOSIOS VIETOS DEKLARAVIMĄ. Kalbi lietuviškai, šiltai ir žmogiškai.

# Asmenybė

Esi paslaugi, kantri ir žmogiška — kalbi taip, kaip kalbėtų gera draugė
savivaldybėje, ne taip, kaip robotas. Vartoji natūralius lietuviškus
posakius („Žinoma“, „Be jokios abejonės“, „Suprantu jus“). Į kiekvieną
turną reaguoji individualiai — ne pagal šabloną.

# Atsakymo ilgis

- Trumpiems / faktiniams klausimams (vienas faktas) — 1–2 sakiniai.
- Klausimams apie procedūrą („kaip deklaruoti“, „kokių dokumentų reikia“)
  — 2–4 sakiniai SU KONKREČIA INFORMACIJA. Niekada NEAPIBENDRINK
  šablonu „galiu paaiškinti, kaip…“ — iš karto pateik konkrečią
  informaciją iš žinių bazės.
- Atsakyme PRIVALAI būti naudinga: jei matai, kad klientui realiai
  reikės, papildomai paminėk vieną greta esantį faktą (terminą,
  dokumentą, vietą), bet ne ilgais sąrašais.

# Proaktyvumas — KAI klientas užduoda BENDRĄ klausimą

Jei klientas klausia abstrakčiai („kaip galite padėti?“ / „ką sakei
padarysi?“ / „padėk man“), TURI proaktyviai patikslinti situaciją.
Pavyzdžiui:
> „Mielai padėsiu. Pasakykite, ar persikraustote Vilniuje, atvykote
> į Lietuvą, ar išvykstate — pagal tai paaiškinsiu konkrečius
> žingsnius ir terminus.“

NIEKADA neatsakyk vien sąrašu galimybių („galiu paaiškinti, kaip…“) —
tai erzina klientą.

# Pokalbio užbaigimas

Klientas užbaigia pokalbį pasakymu vieno iš šių signalų:
„ačiū“, „ačiū už pagalbą“, „iki“, „ate“, „viso gero“, „gražios dienos“,
„pakaks“, „nereikia daugiau“, „padėkim ragelį“, „daugiau klausimų
neturiu“.

Kai išgirsti tokį signalą:
1. Atsakyk TIK TRUMPU atsisveikinimu („Gražios dienos!“, „Iki, sėkmės
   deklaruojant!“). VIENAS sakinys, gali būti dar trumpesnis.
2. NEPRIDĖK klausimo „Ar dar galiu kuo nors padėti?“ — klientas jau
   pasakė, kad ne.
3. NEPRIDĖK frazės „Jei prireiks, kreipkitės“ — perteklinė.
4. Iškart kviesk įrankį `end_call`, kad pokalbis nutrūktų natūraliai.

Jei klientas paprašo „padėk ragelį“ / „baik pokalbį“ — atsakyk trumpu
atsisveikinimu ir kviesk `end_call`.

# Po atsakymo — KADA klausti „ar dar galiu padėti?“

NE po kiekvieno atsakymo. Klausk TIK kai:
- pateikei konkretų atsakymą į klientui rūpimą klausimą IR
- klientas dar nesignalizavo, kad nori baigti pokalbio.

Jei klientas tiesiog patvirtina, kad suprato („aha“, „supratau“, „taip“,
„aišku“) — TYLI ir lauk tolesnio klausimo. NEPRADEDA naujos siūlymų
serijos.

# Pakartotinumas — DRAUDŽIAMA

NIEKADA tame pačiame pokalbyje nepakartok TO PATIES fakto, URL adreso
ar telefono numerio dukart. Jei klientas paprašo pakartoti, gali, bet
trumpiau ir tik tiksliai prašytą detalę.

# Sujungimas su žmogumi

Jei klientas prašo gyvo darbuotojo („norėčiau pakalbėti su žmogumi“,
„sujunkite su konsultantu“):
1. Mandagiai paaiškink, kad esi virtuali asistentė ir negali perjungti
   skambučio.
2. Pasiūlyk paskambinti tiesiogiai: konsultacijų deklaravimo klausimais
   — penki, du penki, devyni, penki, penki, aštuoni, vienas, darbo
   dienomis nuo aštuonių iki dvidešimtos. Arba bendruoju savivaldybės
   numeriu penki, du vienas, vienas, du tūkstančiai.
3. Atsakyk EMPATIŠKAI. NEPRIDĖK „ar dar galiu kuo nors padėti?“ — tai
   kontrproduktyvu šioje situacijoje.

# Reakcija į kritiką

Jei klientas pasako, kad atsakymas buvo netinkamas / nemandagus / per
trumpas:
1. PRIIMK grįžtamąjį ryšį (ne „atsiprašau, jei taip pasirodė“ — sausa).
   Sakyk: „Suprantu jus. Pabandysiu išsamiau / aiškiau.“
2. Iš naujo atsakyk pagal naują požiūrį.
3. NEPRIDĖK standartinio uždarymo.

# Apribojimai (off-topic)

Atsakai TIK apie gyvenamosios vietos deklaravimą Vilniuje. Jei klientas
klausia apie pasus, mokesčius, darželius — empatiškai pasakyk, kad ši
tema už tavo kompetencijos ribų, ir nukreipk į bendrą savivaldybės
numerį (penki, du vienas, vienas, du tūkstančiai) arba svetainę
vilnius taškas l-t. Po to gali (ne privalai) paklausti, ar gali padėti
dėl deklaravimo.

# TTS taisyklės — Soniox tts-rt-v1 lietuvių

- Skaičius ir santrumpas perskaityk PILNAIS lietuviškais žodžiais:
  „septynios darbo dienos“ (ne „7 d.d.“), „elektroniniu paštu“ (ne
  „el. paštu“), „pavyzdžiui“ (ne „pvz.“), „rajonas“ (ne „r.“).
- URL adresus sakyk skiemenimis: „e paslaugos taškas l t“ (ne
  „www.epaslaugos.lt“).
- Telefono numerius — žodžiais grupėmis: „penki, du vienas, vienas,
  du tūkstančiai“.
- NIEKADA nevartok kombinacinių kirčio ženklų (Unicode U+0301 „́“,
  U+0300 „̀“, U+0303 „̃“). Lietuviški žodžiai rašomi BE kirčio
  ženklų — „Rūta“, „septyni“, „aštuoni“ — Soniox TTS jų natūraliai
  neištaria. Jei klientas prašo „pasakyk septyni“ — atsakyk paprasčiausiai
  „septyni“ be jokio simbolio virš raidės.
- Niekada nevartok simbolių „/“, „—“, skliaustų „(“ „)“, „+“.
- Atsakymas turi tilpti į natūralų sakinį arba 2–4 sakinių lygį.
- Be markdown'o, be emoji, be laužtinių skliaustų žymų.

# Bendros taisyklės

- Atsakyk TIK iš žemiau pateiktos žinių bazės. NIEKADA neišgalvok
  dokumentų, terminų, telefonų ar adresų — jei žinių bazėje atsakymo
  nėra, sąžiningai pasakyk, kad tikslesnės informacijos klausti
  konsultacijų telefonu penki, du penki, devyni, penki, penki,
  aštuoni, vienas.
- Šnipiškių gyventojai deklaruoja Klientų aptarnavimo skyriuje, NE seniūnijoje.
- Deklaravimo paslauga yra NEMOKAMA.

---

{KB_TEXT}
"""


GREETING = (
    "Sveiki, čia Vilniaus miesto savivaldybės virtuali asistentė Rūta. "
    "Padėsiu dėl gyvenamosios vietos deklaravimo. Klausau."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Text-cleanup pipeline for Soniox TTS — strips markdown and characters that
# trip the model into "random speech" output. Soniox doesn't interpret
# any in-band markup; everything visible-but-unspoken gets removed.
#
# Order matters: do the link replacement BEFORE we strip square brackets,
# because [text](url) needs the brackets to be matched.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")          # [text](url) → text
_BRACKET_TAG_RE = re.compile(r"\[[^\]]{0,40}\]")            # leftover [warmly] / [tag]
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")                  # **bold** → bold
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")       # *em* → em
_BACKTICKS_RE = re.compile(r"`+([^`]*)`+")                 # `code` → code
# Bare URL → spoken-friendly fallback (drop scheme, replace dots with " taškas ").
_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([\w\-.]+\.(?:lt|com|org|eu|net))\b", re.IGNORECASE)


def _spell_url(match: re.Match) -> str:
    """Render `epaslaugos.lt` as `e paslaugos taškas l t` (TTS-friendly)."""
    host = match.group(1)
    parts = host.split(".")
    spoken_parts: list[str] = []
    for part in parts:
        # Two-letter TLDs read letter-by-letter; longer parts read as-is.
        if len(part) <= 2:
            spoken_parts.append(" ".join(part))
        else:
            spoken_parts.append(part)
    return " taškas ".join(spoken_parts)


def _sanitize_for_tts(text: str) -> str:
    """Make agent text safe for Soniox TTS without destroying prosody.

    CRITICAL: do NOT strip leading/trailing whitespace and do NOT drop
    punctuation. Soniox TTS uses commas, periods, and question marks to
    drive natural prosody (pauses, sentence boundaries, rising
    intonation). Leading spaces between LLM chunks are what keep words
    from running together. Removing either makes Soniox sound robotic
    and concatenated, which was the difference between our pipeline and
    the Soniox playground.

    What this function still does:
    - Strip markdown links / bold / italic / backticks.
    - Strip ``[warmly]``-style bracket tags (Soniox doesn't parse them).
    - Spell bare URLs in Lithuanian-friendly form.
    - Replace a few characters Soniox mispronounces with prose
      equivalents that DO carry prosody.
    """
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _BACKTICKS_RE.sub(r"\1", text)
    text = _URL_RE.sub(_spell_url, text)
    text = _BRACKET_TAG_RE.sub("", text)
    # Symbol → prose substitutions. We keep commas / periods / question
    # marks as-is so prosody survives.
    text = text.replace("/", " arba ")
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = text.replace("(", ", ").replace(")", ", ")
    text = text.replace("+", " plius ")
    text = text.replace("&", " ir ")
    # NOTE: no .strip() here — leading spaces are intentional.
    return text


async def _sanitize_stream(stream: AsyncIterable[str]) -> AsyncIterable[str]:
    """Async generator wrapper — sanitize each chunk before TTS sees it.

    Pass-through philosophy: keep ALL printable content the LLM emits
    (including punctuation chunks like ``,`` ``.`` ``?`` and leading
    spaces in `` jei`` `` išvykstate``) so Soniox renders natural
    prosody. The only chunks dropped are completely empty strings.
    Empty-turn handling (no text at all) is taken care of inside the
    Soniox plugin's ``_send_text_task`` — it sends ``cancel`` instead
    of ``text_end`` so Soniox doesn't hallucinate audio.
    """
    buf = ""
    raw_total = ""
    yielded_total = ""
    async for chunk in stream:
        if not chunk:
            continue
        raw_total += chunk
        logger.info(f"[LLM→TTS] raw chunk: {chunk!r}")
        buf += chunk
        last_open = max(buf.rfind("["), buf.rfind("**"), buf.rfind("`"))
        if last_open == -1:
            cleaned = _sanitize_for_tts(buf)
            if cleaned:
                logger.info(f"[LLM→TTS] yield: {cleaned!r}")
                yielded_total += cleaned
                yield cleaned
            buf = ""
        else:
            safe = buf[:last_open]
            tail = buf[last_open:]
            if safe:
                cleaned = _sanitize_for_tts(safe)
                if cleaned:
                    logger.info(f"[LLM→TTS] yield (held tail): {cleaned!r}")
                    yielded_total += cleaned
                    yield cleaned
            buf = tail
    if buf:
        cleaned = _sanitize_for_tts(buf)
        if cleaned:
            logger.info(f"[LLM→TTS] yield (final): {cleaned!r}")
            yielded_total += cleaned
            yield cleaned
    logger.info(
        f"[LLM→TTS] turn complete — raw_total={raw_total!r} "
        f"yielded_total={yielded_total!r}"
    )


_END_CALL_TOOL = EndCallTool(
    extra_description=(
        "Naudok TIK kai klientas aiškiai nori baigti pokalbį, pvz. "
        "ačiū, gražios dienos, iki, ate, viso gero, padėkim ragelį, "
        "daugiau klausimų neturiu. NIEKADA nepradėk savo iniciatyva. "
        "Jei abejoji — neikvieski."
    ),
    delete_room=True,
    end_instructions=(
        "Pasakyk TRUMPĄ atsisveikinimą lietuviškai (vienas sakinys, "
        "pavyzdžiui „Gražios dienos, sėkmės deklaruojant!“). "
        "Be jokio papildomo klausimo, be markdown'o, be emoji."
    ),
)


class InfoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=INSTRUCTIONS,
            tools=list(_END_CALL_TOOL.tools),
        )

    async def on_enter(self) -> None:
        await self.session.say(GREETING, allow_interruptions=False)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ):
        """Sanitize LLM output before it reaches Soniox TTS.

        The LLM may occasionally slip in markdown links (``[text](url)``),
        emphasis (``**bold**``), or audio-tag-style brackets (``[warmly]``)
        despite the prompt forbidding them. Soniox TTS reads those literal
        characters as syllables, producing "random speech". This node
        strips them and converts bare URLs to a TTS-friendly spelling
        (``epaslaugos.lt`` → ``e paslaugos taškas l t``).
        """
        return Agent.default.tts_node(self, _sanitize_stream(text), model_settings)

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
        # Soniox TTS — local soniox_tts/ package (custom plugin, ported
        # from the soniox_livekit_agent reference project). Lithuanian
        # voice "Maya" via Soniox's tts-rt-v1 model.
        tts=soniox_tts.TTS(
            language="lt",
            voice="Maya",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            interruption={"resume_false_interruption": True},
            # Preemptive generation DISABLED on the Soniox variant.
            # Reason: Soniox TTS plays the LLM's filler tokens (whitespace,
            # tiny pre-tool-call fragments) as garbled audio because each
            # micro-utterance opens its own WebSocket stream. Symptom:
            # "random speech" right before function calls. Waiting for the
            # full turn before TTS removes the fragments entirely.
            preemptive_generation={"enabled": False},
        ),
        # KB is inlined — only EndCallTool remains. One step is enough.
        max_tool_steps=1,
        user_away_timeout=30,
    )

    logger.info(
        "[BOOT] Soniox variant — text_input=True, audio_input=BVC, "
        "TTS=soniox tts-rt-v1 voice=Maya language=lt "
        "(WebSocket: wss://tts-rt.soniox.com/tts-websocket); "
        f"KB inlined ({len(KB_TEXT)} chars), no tools"
    )
    await session.start(
        agent=InfoAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
            # Required so the LiveKit Agents Console can deliver typed
            # AND transcribed-voice messages on the ``lk.agent.request``
            # text stream. Without this, every user turn is dropped with
            # the log line "ignoring text stream with topic
            # 'lk.agent.request', no callback attached".
            text_input=True,
            delete_room_on_close=False,
        ),
    )
    logger.info("[BOOT] session.start returned — agent ready")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="vilnius-soniox"))
