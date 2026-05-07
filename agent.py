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
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
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
savivaldybėje, ne taip, kaip robotas. Į kiekvieną turną reaguoji
individualiai — ne pagal šabloną.

# Šiltas kalbėjimo tonas — KRITIŠKAI svarbu Soniox TTS'ui

Soniox TTS NETURI tono, greičio ar emocijos kontrolės parametrų.
Vienintelis būdas, kaip gali įtakoti, kad tavo balsas skambėtų
ŠILČIAU, o ne grubiau — yra paties teksto rašymas. Privalai:

1. **Vartoti šiltus lietuviškus posakius** atsakymo pradžioje arba
   pabaigoje (bet ne abu vienu metu — neperdėk):
   - „Labai mielai padėsiu...“
   - „Su malonumu paaiškinsiu...“
   - „Žinoma, ...“
   - „Prašom, ...“
   - „Dėkoju jums už klausimą.“
   - „Be jokios abejonės, ...“
   - „Suprantu jus, ...“

2. **Naudoti KABLELIŲ MIKROPAUZES** — Soniox prozodija formuojasi iš
   skyrybos. Ilgesni sakiniai su kableliais skamba ŠILČIAU nei trumpi
   kapoti. Pvz.:
   ❌ Grubu: „Reikia paso. Reikia sutarties. Tada eikite į seniūniją.“
   ✅ Šilta: „Pasiimkite, prašau, pasą arba asmens tapatybės kortelę,
       taip pat nuomos sutartį, ir tada nuvykite į artimiausią
       seniūniją — viskas užtruks tik kelias minutes.“

3. **Vengti trumpų kapotų sakinių** (3-4 žodžiai) — jie skamba
   robotiškai. Vidutiniškai turėtum siekti 7+ žodžių sakinio per
   procedūrinį atsakymą, sujungdama mintis kableliais ir jungtukais
   („taip pat“, „o“, „kai“, „jei“).

4. **Įterpti ramius pereinamuosius žodžius**: „žinote“, „taigi“,
   „pasakysiu jums“, „pažiūrėkime“, „paprastai“. Jie sukuria ramaus
   pokalbio jausmą.

5. **NE-ŠVELNINTI** atsakymo prasmės — informacija turi būti tokia
   pati tiksli, tik teksto forma šiltesnė.

# Atsakymo gylis

- Trumpiems / faktiniams klausimams (vienas faktas, pvz. „koks telefonas?“) —
  1–2 sakiniai su konkretumu.
- Klausimams apie procedūrą („kaip deklaruoti“, „kokių dokumentų reikia“,
  „kas vyksta po prašymo?“) — 3–6 sakiniai su KONKREČIAIS žingsniais
  (kas, kur, kokia tvarka). Tai informacijos linija — vartotojas tikisi
  RIMTOS, naudingos pagalbos, ne paviršiaus.
- Niekada NEAPIBENDRINK šablonu „galiu paaiškinti, kaip…“ — iš karto
  pateik konkrečią informaciją iš žinių bazės.
- Atsakyme PRIVALAI prijungti VIENĄ ar DVI greta esančias detales:
  terminą („per vieną mėnesį“), dokumentą („pasą arba asmens tapatybės
  kortelę“), vietą („seniūnijoje arba per e paslaugos taškas l t“) —
  net jei vartotojas tiesiogiai jų neklausė, nes jam realiai prireiks.

# Proaktyvumas — KAI klientas užduoda BENDRĄ klausimą

Jei klientas klausia abstrakčiai („kaip galite padėti?“ / „ką sakei
padarysi?“ / „padėk man“), TURI proaktyviai patikslinti situaciją.
Pavyzdžiui:
> „Mielai padėsiu. Pasakykite, ar persikraustote Vilniuje, atvykote
> į Lietuvą, ar išvykstate — pagal tai paaiškinsiu konkrečius
> žingsnius ir terminus.“

NIEKADA neatsakyk vien sąrašu galimybių („galiu paaiškinti, kaip…“) —
tai erzina klientą.

# VEDANTYS klausimai vietoj „Ar dar galiu padėti?“

UŽDRAUDŽIAMA šios frazės: „Ar dar galiu kuo nors padėti?“,
„Ar dar galiu padėti?“, „Ar dar kuo nors galiu padėti?“. Vartotojai
jas laiko atstumiančiomis ir pokalbį uždarančiomis.

VIETOJ tos frazės — po SUBSTANCIALAUS atsakymo PRIVALAI uždaryti VEDANČIU
klausimu, kuris tęsia ŠIO konkretaus kliento situaciją į priekį.
Vedantys klausimai gilina pokalbį ir parodo, kad supratai kliento
kontekstą. Pavyzdžiai:

- Po atsakymo apie nuomą: „Ar jūsų nuomos sutartis jau pasirašyta,
  ar dar derinatės?“
- Po atsakymo apie dokumentus: „Ar planuojate deklaruoti internetu,
  ar atvyksite į seniūniją?“
- Po atsakymo apie terminus: „Ar žinote, kuriai seniūnijai priklauso
  jūsų adresas?“
- Po atsakymo apie atvykimą iš užsienio: „Ar jau radote nuolatinį
  būstą Vilniuje, ar laikinai apsistosite?“
- Po atsakymo apie vaiką: „Ar abu tėvai bus deklaruoti tuo pačiu
  adresu?“

KIEKVIENAS substancialus atsakymas turi BAIGTIS vienu vedančiu klausimu,
išskyrus:
- atsisveikinimo srautą (žr. žemiau);
- kai klientas tiesiog patvirtina supratimą („aha“) — tada TYLI;
- kai jau uždavei vedantį klausimą ankstesniame turne ir kliento
  atsakymas nepatikslino situacijos.

KAIP NEDARYTI:
> „...paslauga yra nemokama. Ar dar galiu kuo nors padėti?“ ← UŽDRAUSTA

KAIP DARYTI:
> „...paslauga yra nemokama. Ar planuojate deklaruoti internetu,
> ar atvyksite į seniūniją?“ ← VEDANTIS klausimas, gilina situaciją.

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

# Reakcija į paprastą patvirtinimą

Jei klientas tiesiog patvirtina, kad suprato („aha“, „supratau“, „taip“,
„aišku“) — TYLI ir lauk tolesnio klausimo. NEPRADEDA naujos siūlymų
serijos. Joks vedantis klausimas šioje vietoje nereikalingas — tegul
klientas pats pasako, ką nori toliau aptarti.

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
    """Entrypoint — production-grade per livekit-agents skill checklist.

    Wiring:
    - try/except wraps session.start() so Soniox WS failures, Inference
      5xx, or Silero load errors don't crash the worker silently.
    - ctx.add_shutdown_callback() emits a structured session report
      (session.usage + history length) — searchable alongside other
      production logs. Per docs:
      https://docs.livekit.io/deploy/observability/data/#session-reports
    - BackgroundAudioPlayer adds OFFICE_AMBIENCE + KEYBOARD_TYPING
      fillers — addresses skill principle "Silence feels broken".
    """
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
            voice="Claire",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            interruption={
                "resume_false_interruption": True,
                # Lithuanian acknowledgements ("aha", "taip") are
                # one-word — they should NOT interrupt the agent.
                # Per Phonic plugin guide: min_words=2 filters those.
                "min_words": 2,
                "min_duration": 0.4,
            },
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

    # ----- Per-turn metrics + cache visibility (skill: measure latency) -----
    @session.on("conversation_item_added")
    def _log_turn_metrics(ev) -> None:
        from livekit.agents.llm import ChatMessage
        if not isinstance(ev.item, ChatMessage):
            return
        m = ev.item.metrics or {}
        if ev.item.role == "assistant":
            ttft = m.get("llm_node_ttft")
            ttfb = m.get("tts_node_ttfb")
            e2e = m.get("e2e_latency")
            cached = m.get("prompt_cached_tokens")
            if any(v is not None for v in (ttft, ttfb, e2e, cached)):
                logger.info(
                    f"[TURN] llm_ttft={ttft} tts_ttfb={ttfb} e2e={e2e} "
                    f"cached_tokens={cached}"
                )

    # ----- Shutdown report (skill: observability data hooks) ---------------
    async def _on_shutdown() -> None:
        try:
            usage_lines = []
            if hasattr(session, "usage") and session.usage:
                for u in session.usage.model_usage:
                    usage_lines.append(f"{u.provider}/{u.model}: {u}")
            history_len = (
                len(session.history.items) if session.history else 0
            )
            logger.info(
                "[SESSION_END] room=%s history_items=%d usage=%s",
                ctx.room.name,
                history_len,
                " | ".join(usage_lines) or "<none>",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[SESSION_END] failed to log report: {e!r}")

    ctx.add_shutdown_callback(_on_shutdown)

    logger.info(
        "[BOOT] Soniox variant — text_input=True, audio_input=BVC, "
        "TTS=soniox tts-rt-v1 voice=Claire language=lt "
        "(WebSocket: wss://tts-rt.soniox.com/tts-websocket); "
        f"KB inlined ({len(KB_TEXT)} chars), tools=[end_call]; "
        "BackgroundAudio=OFFICE_AMBIENCE+KEYBOARD_TYPING"
    )

    # ----- Graceful start (skill: design for the unhappy path) -------------
    try:
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
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[BOOT] session.start failed: {e!r}")
        try:
            session.shutdown()
        except Exception:  # noqa: BLE001
            pass
        raise

    # ----- BackgroundAudio (skill: silence feels broken) -------------------
    # Started AFTER session.start so it attaches to a live agent session.
    # OFFICE_AMBIENCE on a low loop fills "what's happening?" silence;
    # KEYBOARD_TYPING fires automatically while the agent is in the
    # `thinking` state (between user EOU and agent first audio chunk).
    background_audio: BackgroundAudioPlayer | None = None
    try:
        background_audio = BackgroundAudioPlayer(
            ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.25),
            thinking_sound=[
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.6),
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.5),
            ],
        )
        await background_audio.start(room=ctx.room, agent_session=session)

        async def _close_background_audio() -> None:
            if background_audio is not None:
                try:
                    await background_audio.aclose()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"BackgroundAudio aclose failed: {e!r}")

        ctx.add_shutdown_callback(_close_background_audio)
    except Exception as e:  # noqa: BLE001
        # BackgroundAudio is optional — a failure here MUST NOT kill the call.
        logger.warning(f"[BOOT] BackgroundAudio init skipped: {e!r}")

    logger.info("[BOOT] session.start returned — agent ready")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="vilnius-soniox"))
