"""Vilniaus miesto savivaldybės gyvenamosios vietos deklaravimo DUK.

Šaltinis (verbatim): https://paslaugos.vilnius.lt/service/Gyvenamosios-vietos-deklaravimas
Surinkta 2026-05-04 — 15 klausimų-atsakymų.

Šis modulis yra:
1. Žinių bazė ``InfoAgent`` ``lookup_faq`` įrankiui.
2. Kanoninis testų korpusas ``tests/test_faq_behavioral.py`` —
   kiekvienas DUK tampa elgsenos testu (žr. plano tests skyrių).

Sluoksniai (slug'ai) yra trumpi ir lietuviški, kad LLM galėtų juos
parinkti iš sistemos prompt'o pagal kliento klausimo temą.
"""
from __future__ import annotations

from typing import TypedDict


class FAQEntry(TypedDict):
    question: str
    answer: str


# Kanoninis 15 DUK klausimų-atsakymų sąrašas. Tekstas — verbatim iš
# paslaugos.vilnius.lt; nieko neperfrazuoju, kad atsakymas būtų tikslus
# ir testai būtų deterministiški.
FAQS: dict[str, FAQEntry] = {
    "persikrausciau_i_busta": {
        "question": "Ką turėčiau žinoti, jei persikėliau į būstą Vilniuje?",
        "answer": (
            "Jeigu planuojate tame būste gyventi, rekomenduojama ten ir deklaruoti "
            "gyvenamąją vietą. Tai galima padaryti elektroniniu būdu "
            "www.epaslaugos.lt arba nuvykti į seniūniją ar savivaldybę. Jei būstą "
            "nuomojatės, pakaks asmens dokumento ir būsto nuomos arba panaudos "
            "sutarties. Būsite deklaruotas sutarties galiojimo laikotarpiui. "
            "Pasibaigus šiam terminui, deklaravimas automatiškai pasinaikins."
        ),
    },
    "apskaita_iki_2015": {
        "question": (
            "Buvau įtrauktas į gyvenamosios vietos neturinčių asmenų apskaitą "
            "prie Vilniaus miesto savivaldybės (tai atlikau iki 2015-12-31). "
            "Ką turiu daryti?"
        ),
        "answer": (
            "Turite suskubti deklaruoti gyvenamąją vietą, kadangi, jei to dar "
            "nepadarėte, nuo 2018-01-01 įrašai apie Jūsų įtraukimą į apskaitą "
            "Gyventojų registre nustojo galioti ir Jus likote be deklaravimo "
            "vietos. Tai gali Jums sutrukdyti gauti tam tikras paslaugas, "
            "kurias suteikiant privalomas gyvenamosios vietos deklaravimas, "
            "pavyzdžiui, prarasite prioritetus, registruojant vaiką į darželius, "
            "negalėsite registruoti automobilio savo vardu, neteksite teisės į "
            "socialines išmokas ir kita."
        ),
    },
    "gedimino_pr_24_40": {
        "question": (
            "Buvau prisideklaravęs adresu Gedimino pr. 24-40, Vilniuje. "
            "Ar nuo 2018-01-01 bus panaikinta mano deklaracija?"
        ),
        "answer": (
            "Ne, nebus. Bet, kadangi deklaruoti gyvenamąją vietą tapo paprasčiau, "
            "siūlome, kai tik atsiras galimybė, deklaruoti savo faktinę "
            "gyvenamąją vietą."
        ),
    },
    "kaip_deklaruoti": {
        "question": "Kokiu būdu galėčiau deklaruoti savo gyvenamąją vietą?",
        "answer": (
            "Internetu www.epaslaugos.lt, nuvykus į seniūniją arba savivaldybę."
        ),
    },
    "tapatybes_dokumentai": {
        "question": (
            "Kokius asmens tapatybę patvirtinančius dokumentus reikia turėti, "
            "norint deklaruoti gyvenamąją vietą?"
        ),
        "answer": (
            "Reikalingas galiojantis asmens tapatybę patvirtinantis dokumentas "
            "(asmens tapatybės kortelė arba pasas). Deklaruojant nepilnamečio "
            "vaiko gyvenamąją vietą, reikia turėti ir vaiko gimimo liudijimą "
            "(arba jo išrašą) ir arba pasą (jei toks pagamintas)."
        ),
    },
    "savininko_sutikimas_butas": {
        "question": (
            "Ar deklaruojant naują gyvenamąją vietą ne savo bute, "
            "reikia buto savininko sutikimo?"
        ),
        "answer": (
            "Jeigu turite nuomos arba panaudos sutartį, kurioje būsto savininkas "
            "jau nurodė, kad galite tam tikrą laiką gyventi jo būste, tuomet "
            "deklaravimo įstaigai (seniūnijai arba savivaldybei) reikia pateikti "
            "nuomos arba panaudos sutartį. Raštiško savininko leidimo Jums jau "
            "nebereikės. Nuomos sutartis neturi būti patvirtinta notaro (nėra "
            "tokio reikalavimo). Joje turi būti aiškiai nurodyta, jog buto "
            "savininkas leidžia gyventi arba nuomotis būstą tam tikriems "
            "asmenims. Sutartis turi būti pasirašyta abiejų pusių."
        ),
    },
    "savininko_sutikimas_vaikas": {
        "question": (
            "Ar reikia buto savininko sutikimo, norint kartu deklaruoti "
            "savo nepilnametį vaiką?"
        ),
        "answer": (
            "Ne. Deklaruojant nepilnamečius vaikus, atskiro buto savininko "
            "sutikimo nereikia, nes vaikai deklaruojami prie tėvų (įtėvių, "
            "globėjų arba rūpintojų). Tai galite atlikti elektroniniu būdu "
            "www.epaslaugos.lt arba nuvykę į deklaravimo įstaigą, "
            "tai yra seniūniją arba savivaldybę."
        ),
    },
    "studentas_bendrabutyje": {
        "question": "Kaip studentui deklaruoti gyvenamąją vietą universiteto bendrabutyje?",
        "answer": (
            "Turite deklaravimo įstaigai (seniūnijai arba savivaldybei) pateikti "
            "universiteto vadovybės (arba bendrabučio valdytojo) raštišką sutikimą."
        ),
    },
    "negyvenamoji_paskirtis": {
        "question": (
            "Jeigu būsto paskirtis yra negyvenamoji, "
            "ar galima deklaruotis kitos paskirties patalpose?"
        ),
        "answer": (
            "Gyventojas deklaruoja savo gyvenamąją vietą, kurioje praleidžia "
            "daugiausiai laiko arba yra labiausiai su ja susijęs. Jeigu gyvenate "
            "kitos paskirties būste, tuomet ją galite deklaruoti kaip savo "
            "gyvenamąją vietą."
        ),
    },
    "sodo_namelis": {
        "question": "Gyvenu sodo namelyje, ar galiu jame deklaruotis?",
        "answer": (
            "Galite, jei sodo namelis įregistruotas ir turi adresą. Tokiu atveju, "
            "galite deklaruotis elektroniniu būdu www.epaslaugos.lt arba nuvykus "
            "su asmens dokumentu į seniūniją arba savivaldybę."
        ),
    },
    "viesbutis": {
        "question": (
            "Ar galiu deklaruoti savo gyvenamąją vietą viešbutyje ar svečių "
            "namuose, kuriuose esu apsistojęs ilgesnį laiką?"
        ),
        "answer": (
            "Gyvenant svečių namuose arba viešbutyje, Jūs taip pat turite teisę, "
            "gavę būsto savininko leidimą, deklaruoti savo gyvenamąją vietą."
        ),
    },
    "zemes_sklypas": {
        "question": "Ar galiu deklaruotis žemės sklype, jei sklypas turi adresą (statinių nėra)?",
        "answer": "Ne, negalima.",
    },
    "kiek_asmenu_buste": {
        "question": "Kiek asmenų gali deklaruoti savo gyvenamąją vietą viename būste?",
        "answer": "Būsto ploto, tai yra kvadratūros, apribojimo nėra.",
    },
    "prie_savivaldybes": {
        "question": "Ar galima deklaruotis prie Vilniaus miesto savivaldybės?",
        "answer": (
            "Tikslus šios paslaugos pavadinimas yra įtraukimas į gyvenamosios "
            "vietos nedeklaravusių asmenų apskaitą prie Vilniaus miesto "
            "savivaldybės. Į šią apskaitą ribotam terminui, jeigu jie nėra "
            "deklaravę gyvenamosios vietos, gali būti įtraukiami tik benamiai, "
            "taip pat asmenys, palikę vaikų globos namus ar šeimyną, atliekantys "
            "bausmę pataisos įstaigose ar laikomi tardymo izoliatoriuose, bei "
            "tie, kurie teismo sprendimu gydomi specializuotose psichikos "
            "sveikatos įstaigose, taip pat asmenys, kurie ne nuo jų priklausančių "
            "priežasčių negali deklaruoti savo gyvenamosios vietos ir pateikia "
            "informaciją arba duomenis, pagrindžiančius jų ekonominius, "
            "socialius ar asmeninius interesus toje savivaldybėje. "
            "Kiti asmenys turi deklaruoti faktinę gyvenamąją vietą."
        ),
    },
    "vaikai_prie_savivaldybes": {
        "question": "Ar galima deklaruoti nepilnamečius vaikus prie Vilniaus miesto savivaldybės?",
        "answer": (
            "Vaikai deklaruojami prie tėvų (įtėvių, globėjų arba rūpintojų). "
            "Jeigu abu tėvai (įtėviai, globėjai arba rūpintojai) yra įtraukti į "
            "gyvenamosios vietos nedeklaravusių asmenų apskaitą arba nepilnamečio "
            "gyvenamoji vieta nustatyta su tuo iš tėvų, kuris įtrauktas į šią "
            "apskaitą, vaiko įtraukimas į gyvenamosios vietos neturinčių asmenų "
            "apskaitą galimas."
        ),
    },
}


def faq_index() -> str:
    """Trumpas FAQ indeksas, įterpiamas į sistemos prompt'ą.

    LLM iš šio sąrašo parenka slug'ą ir kviečia ``lookup_faq(faq_id=slug)``,
    kad gautų pilną kanoninį atsakymą.
    """
    lines = []
    for slug, entry in FAQS.items():
        lines.append(f"- {slug}: {entry['question']}")
    return "\n".join(lines)


def get_faq(slug: str) -> FAQEntry | None:
    return FAQS.get(slug)
