"""Struktūrinė Vilniaus deklaravimo žinių bazė.

Skirta įrankiams ``get_required_documents``, ``get_deadline``,
``get_contact_info``, ``get_seniunijos_info`` ir ``get_service_overview``.
Visi tekstai — lietuvių kalba, paruošti perskaityti balsu (be simbolių
``/``, ``+``, ``el.`` ir kitų lietuviškų trumpinių, kurie nesuprantamai
skamba TTS).

Šaltinis: https://paslaugos.vilnius.lt/service/Gyvenamosios-vietos-deklaravimas
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Paslaugos aprašymas (verbatim iš VMSA paslaugos puslapio)
# ---------------------------------------------------------------------------

SERVICE_OVERVIEW: str = (
    "Deklaruoti savo gyvenamąją vietą galima ir internetu, „Elektroninių "
    "valdžios vartų“ interneto svetainėje www.epaslaugos.lt, "
    "arba nuvykus į seniūniją ar savivaldybę. "
    "Gyvenamosios vietos deklaravimo duomenų tvarkymo funkcija perduota "
    "vykdyti Vilniaus miesto savivaldybės administracijos seniūnijoms ir "
    "Klientų aptarnavimo skyriui. "
    "Lietuvos Respublikos gyventojas vienu metu deklaruoja tik vieną "
    "gyvenamąją vietą, net jei jis gyvena keliose vietose."
)


# ---------------------------------------------------------------------------
# Terminai (deadline'ai) pagal scenarijų
# ---------------------------------------------------------------------------

DEADLINES: dict[str, str] = {
    "persikraustymas_lt": (
        "Asmenys, privalantys deklaruoti gyvenamąją vietą, ją deklaruoja "
        "per vieną mėnesį nuo pareigos deklaruoti gyvenamąją vietą atsiradimo."
    ),
    "atvykimas_i_lt": (
        "Atvykus gyventi į Lietuvos Respubliką, gyvenamoji vieta turi būti "
        "deklaruota per vieną mėnesį nuo atvykimo."
    ),
    "isvykimas_iz_lt": (
        "Asmenys, išvykstantys iš Lietuvos Respublikos ilgesniam nei šešių "
        "mėnesių laikotarpiui, gyvenamosios vietos pakeitimą privalo "
        "deklaruoti per septynias darbo dienas iki išvykimo."
    ),
    "be_pakeitimo": (
        "Lietuvos Respublikos gyventojai, kurie yra deklaravę gyvenamąją "
        "vietą ir kurių gyvenamoji patalpa nesikeičia pasikeitus buto arba "
        "namo numeriui, gatvės, gyvenamosios vietovės ar teritorijos "
        "administracinio vieneto pavadinimui, gyvenamosios vietos iš naujo "
        "nedeklaruoja."
    ),
}


# ---------------------------------------------------------------------------
# Reikiami dokumentai pagal scenarijų
# ---------------------------------------------------------------------------

REQUIRED_DOCS: dict[str, list[str]] = {
    "persikraustymas_lt": [
        "Prašymas deklaruoti gyvenamąją vietą.",
        "Asmens tapatybę patvirtinantis dokumentas — Lietuvos piliečio "
        "pasas arba asmens tapatybės kortelė.",
        "Dokumentai, patvirtinantys nuosavybės arba kitokio teisėto valdymo "
        "teisę į gyvenamąją patalpą — pavyzdžiui, nuomos arba panaudos sutartis.",
        "Jei būstas ne Jūsų — savininko (arba bendraturčių) sutikimas, "
        "išskyrus atvejį, kai turite nuomos arba panaudos sutartį, kurioje "
        "savininkas leidimą jau yra išreiškęs.",
    ],
    "atvykimas_i_lt": [
        "Prašymas deklaruoti gyvenamąją vietą.",
        "Galiojantis asmens tapatybę patvirtinantis dokumentas — Europos "
        "Sąjungos valstybės narės piliečio leidimas gyventi nuolat arba "
        "galiojantis Europos Sąjungos valstybės narės piliečio pasas, "
        "arba dokumentas, patvirtinantis, kad asmuo įgijo teisę nuolat "
        "gyventi Lietuvos Respublikoje.",
        "Dokumentai, patvirtinantys teisę į gyvenamąją patalpą.",
        "Savininko sutikimas, jei būstas ne Jūsų ir nėra nuomos arba "
        "panaudos sutarties.",
    ],
    "isvykimas_iz_lt": [
        "Prašymas deklaruoti išvykimą iš Lietuvos Respublikos.",
        "Asmens tapatybę patvirtinantis dokumentas.",
        "Pareiškiama išvykimo data ir šalis, į kurią išvykstate.",
    ],
    "vaiko_deklaravimas": [
        "Vaiko gimimo liudijimas arba jo išrašas, "
        "ir arba vaiko pasas (jei toks pagamintas).",
        "Tėvo arba motinos (įtėvio, globėjo, rūpintojo) asmens tapatybės "
        "dokumentas.",
        "Atskiro buto savininko sutikimo nereikia, nes vaikas deklaruojamas "
        "prie tėvų ar globėjų.",
    ],
    "globejas": [
        "Globą arba rūpybą patvirtinantys dokumentai (arba jų patvirtintos "
        "kopijos).",
        "Globėjo arba rūpintojo asmens tapatybę patvirtinantis dokumentas.",
        "Globotinio asmens tapatybę patvirtinantis dokumentas, jei toks yra.",
    ],
    "studentas_bendrabutyje": [
        "Asmens tapatybę patvirtinantis dokumentas.",
        "Universiteto vadovybės arba bendrabučio valdytojo raštiškas sutikimas.",
    ],
}


# ---------------------------------------------------------------------------
# Kontaktai
# ---------------------------------------------------------------------------

CONTACTS: dict[str, str] = {
    "savivaldybes_telefonas": "+370 5 211 2000",
    "savivaldybes_trumpasis": "1664",
    "savivaldybes_elpastas": "savivaldybe@vilnius.lt",
    "savivaldybes_adresas": "Konstitucijos prospektas 3, Vilnius, LT-09601",
    "konsultacijos_telefonas": "+370 5 259 5581",
    "konsultacijos_darbo_laikas": (
        "darbo dienomis nuo aštuonių iki dvidešimtos valandos, "
        "prieššventinėmis dienomis — nuo aštuonių iki aštuonioliktos valandos"
    ),
    "epaslaugos_url": "www.epaslaugos.lt",
    "vilnius_url": "www.vilnius.lt",
    "seniunijos_url": "www.vilnius.lt/seniunijos",
    "paslauga_trukme": "penkios darbo dienos",
}


# ---------------------------------------------------------------------------
# Seniūnijos — Šnipiškių išimtis
# ---------------------------------------------------------------------------

SENIUNIJOS_INFO: str = (
    "Asmenys gyvenamąją vietą deklaruoja seniūnijoje, aptarnaujančioje tą "
    "savivaldybės teritorijos dalį, kurioje jie gyvena. "
    "Šnipiškių seniūnijos gyventojai gyvenamąją vietą deklaruoja Vilniaus "
    "miesto savivaldybės Klientų aptarnavimo skyriuje. "
    "Konkrečios seniūnijos kontaktai pateikiami svetainėje "
    "www.vilnius.lt/seniunijos."
)


# Vilniaus miesto seniūnijos — naudoja STT context (atpažinti vardus pokalbyje).
SENIUNIJOS_VILNIUJE: list[str] = [
    "Antakalnio",
    "Fabijoniškių",
    "Grigiškių",
    "Justiniškių",
    "Karoliniškių",
    "Lazdynų",
    "Naujamiesčio",
    "Naujininkų",
    "Naujosios Vilnios",
    "Panerių",
    "Pašilaičių",
    "Pilaitės",
    "Rasų",
    "Senamiesčio",
    "Šeškinės",
    "Šnipiškių",
    "Verkių",
    "Vilkpėdės",
    "Viršuliškių",
    "Žirmūnų",
    "Žvėryno",
]


# ---------------------------------------------------------------------------
# Konsultacijos kanalai (informaciniam atsakymui)
# ---------------------------------------------------------------------------

KONSULTACIJOS: str = (
    "Konsultacijas dėl gyvenamosios vietos deklaravimo galite gauti: "
    "„Pagalba gyvai“ portale www.epaslaugos.lt darbo dienomis nuo aštuonių "
    "iki dvidešimtos valandos ir nuo aštuonių iki aštuonioliktos "
    "prieššventinėmis dienomis, "
    "arba telefonu 5 2595581 tomis pačiomis valandomis. "
    "Vilniaus miesto savivaldybės bendras telefonas — 5 211 2000."
)


# ---------------------------------------------------------------------------
# Asmenys, nelaikomi pakeitusiais deklaruotą gyvenamąją vietą
# ---------------------------------------------------------------------------

NELAIKOMI_PAKEITUSIAIS: list[str] = [
    "Lietuvos Respublikos diplomatinėse atstovybėse, konsulinėse įstaigose "
    "ir tarptautinėse organizacijose dirbantys piliečiai bei kartu su jais "
    "išvykę šeimos nariai — visą darbo šiose įstaigose laikotarpį.",
    "Asmenys, atliekantys privalomąją karo arba alternatyviąją krašto "
    "apsaugos tarnybą.",
    "Asmenys, kurie gydosi stacionarinėse sveikatos priežiūros įstaigose.",
    "Moksleiviai ir studentai mokymosi laikotarpiu.",
    "Asmenys, atliekantys laisvės atėmimo bausmę ir laikomi kardomojo "
    "kalinimo vietose.",
    "Jūreiviai.",
]


# ---------------------------------------------------------------------------
# Inline KB block — built once at import time, embedded in the system prompt.
# ---------------------------------------------------------------------------

def build_kb_text() -> str:
    """Render the full KB (FAQ + structured data) as plain Lithuanian text.

    The result is inlined into ``InfoAgent.instructions`` so the LLM can
    answer in a single turn without round-tripping to a tool. With LiveKit
    Inference's prompt caching, the ~2,500-token block is cached across
    turns after the first call, so cost stays flat.
    """
    from knowledge.faqs import FAQS

    parts: list[str] = []

    parts.append("# Žinių bazė: gyvenamosios vietos deklaravimas Vilniuje\n")
    parts.append("## Paslaugos aprašymas\n" + SERVICE_OVERVIEW + "\n")

    parts.append("\n## Terminai")
    for slug, text in DEADLINES.items():
        parts.append(f"- **{slug}**: {text}")

    parts.append("\n## Reikiami dokumentai")
    for slug, docs in REQUIRED_DOCS.items():
        parts.append(f"\n**{slug}**:")
        for d in docs:
            parts.append(f"  - {d}")

    parts.append("\n## Kontaktai")
    parts.append(f"- Bendras savivaldybės telefonas: {CONTACTS['savivaldybes_telefonas']} (trumpasis {CONTACTS['savivaldybes_trumpasis']})")
    parts.append(f"- Elektroninis paštas: {CONTACTS['savivaldybes_elpastas']}")
    parts.append(f"- Adresas: {CONTACTS['savivaldybes_adresas']}")
    parts.append(f"- Konsultacijų telefonas (deklaravimo klausimais): {CONTACTS['konsultacijos_telefonas']}")
    parts.append(f"- Konsultacijų darbo laikas: {CONTACTS['konsultacijos_darbo_laikas']}")
    parts.append(f"- Elektroninių paslaugų portalas: {CONTACTS['epaslaugos_url']}")
    parts.append(f"- Seniūnijų sąrašas: {CONTACTS['seniunijos_url']}")
    parts.append(f"- Paslaugos suteikimo trukmė: {CONTACTS['paslauga_trukme']}")

    parts.append("\n## Seniūnijos\n" + SENIUNIJOS_INFO)
    parts.append("Vilniaus miesto seniūnijos: " + ", ".join(SENIUNIJOS_VILNIUJE) + ".")

    parts.append("\n## Konsultacijų kanalai\n" + KONSULTACIJOS)

    parts.append("\n## Nelaikomi pakeitusiais gyvenamąją vietą")
    for x in NELAIKOMI_PAKEITUSIAIS:
        parts.append(f"- {x}")

    parts.append("\n## Dažniausiai užduodami klausimai (DUK)")
    for slug, entry in FAQS.items():
        parts.append(f"\n### {entry['question']}")
        parts.append(entry["answer"])

    return "\n".join(parts)


KB_TEXT: str = build_kb_text()
