"""Prompts of the question pipeline (krok 13): generation, extra rounds, coverage, quality score.

The coverage prompt lists the existing questions with their status (accepted, or dropped and
why), so the model knows which mistakes to avoid.

Every prompt asks for a JSON object {"items": [...]} (JSON mode of every provider). The
templates use <<name>> placeholders filled by str.replace, so the JSON braces in the text
need no escaping. The faithfulness and answerability prompts live in their modules and use
the record texts from here.

Step 13c: no prompt asks for hints or shows them; the hint is built by the code from the
sources of the question (qgen.questions.hints).
"""

from __future__ import annotations

from qgen.config import Settings
from qgen.questions.types import QuestionSettings, type_rules_text

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _fill(template: str, **values) -> str:
    for name, value in values.items():
        template = template.replace(f"<<{name}>>", str(value))
    return template


# --- generation -----------------------------------------------------------------------------

GENERATION_SK = """
Si skúsený učiteľ. Z tematických celkov učebného materiálu a ich textu vytvor otázky a úlohy pre študentov.

HLAVNÁ ÚLOHA:
- Pre každý tematický celok vytvor čo najviac rôznych otázok, ktoré obsah tematického celku dovoľuje – pokry každý pojem, vzorec, postup, príklad a súvislosť z jeho textu.
- Každá otázka sa pýta na niečo iné alebo z iného uhla pohľadu. Tú istú otázku nepreformulúvaj.
- Spodná hranica: tematický celok s učivom nesmie mať menej otázok ako <<per_section>>. Je to iba poistka, nie cieľ – ak obsah dovoľuje viac, vytvor viac.
- Výnimka: iba ak text tematického celku naozaj neobsahuje žiadne učivo, vráť preň namiesto otázok jediný objekt {"lo_id": <id>, "reason": "<krátky dôvod>"}.

OBSAH:
- Otázky vychádzajú iba z textu svojho tematického celku. NEvymýšľaj informácie, ktoré v texte nie sú.
- Zadanie je samostatná vecná otázka o učive (definícia, rozdiel, príčina, postup, vlastnosť, výpočet, použitie).
- Študent text ani dokument nemá. NEpíš „Prečítaj si text…“, „Na základe textu…“ a neodkazuj na strany, kapitoly, obrázky ani časti dokumentu (nie „podľa príkladu na strane 66“). Ak otázka stavia na príklade z textu, uveď v zadaní všetky jeho údaje.
- Zadanie, možnosti a kľúčové slová píš po slovensky.

TYPY OTÁZOK:
<<type_rules>>

PRAVIDLÁ PRE TYPY:
- "choice": v "choices" presne <<choice_options>> možností. Správna je jedna alebo viac možností ("is_correct": true), ostatné majú "is_correct": false. Nesprávne možnosti sú vierohodné a pochádzajú z toho istého učiva (zámena pojmov, častá chyba, chybný výpočet). Nepoužívaj možnosti typu „všetky uvedené“ ani „žiadna z uvedených“. Žiadne dve možnosti nemajú rovnaký text ani význam. Poradie správnych možností striedaj.
- "open": v "choices" práve jedna položka {"text": "<vzorová odpoveď>", "is_correct": true}. Vzorová odpoveď je krátka a presná (pojem, číslo s jednotkou alebo jedna veta), lebo ju učiteľ hodnotí ručne alebo presnou zhodou. Pri praktickej úlohe ("practical": true) je v nej vzorové riešenie s postupom a výsledkom.
- "no_answer": úloha, ktorá sa nevyhodnocuje (napíš esej, zamysli sa, prediskutuj). Zadanie je jasné a súvisí s učivom tematického celku. "choices" je prázdne pole [].

ČASY:
- "time_to_handle" je čas v sekundách, ktorý študent potrebuje na vypracovanie; "time_to_explain" je čas, ktorý učiteľ potrebuje na vysvetlenie riešenia. Odhadni ich podľa náročnosti otázky.
- Rozsahy (time_to_handle / time_to_explain): choice 30–180 / 0–90, open 120–900 / 0–180, no_answer 0–1800 / 0–600.

Tematické celky:
<<lo_blocks>>
<<extra_section>>
Formát výstupu:
Vráť LEN validný JSON objekt {"items": [...]} bez ďalšieho textu. Každá položka poľa "items" má polia:
- "lo_id": id tematického celku, ku ktorému otázka patrí,
- "name": krátky názov otázky (najviac 60 znakov),
- "description": zadanie otázky alebo úlohy,
- "type": "choice", "open" alebo "no_answer",
- "practical": true pri praktickej úlohe (výpočet, postup), inak false,
- "choices": pole objektov {"text": "...", "is_correct": true alebo false} podľa pravidiel typu,
- "time_to_handle": celé číslo (sekundy),
- "time_to_explain": celé číslo (sekundy),
- "bloom_level": úroveň otázky, jedno z: Zapamätať si, Pochopiť, Aplikovať, Analyzovať, Hodnotiť, Vytvoriť,
- "keywords": 1 až 5 kľúčových slov otázky,
- "source": strany, z ktorých otázka vychádza, vo formáte dokument:strana, napr. ["D1:12", "D1:13"].
"""

GENERATION_EN = """
You are an experienced teacher. From the sections of the study material and their text, create questions and tasks for students.

MAIN TASK:
- For each section, create as many different questions as its content allows – cover every concept, formula, procedure, example and relationship in its text.
- Each question asks about something else or from a different angle. Do not rephrase the same question.
- Lower bound: a section with subject matter must not have fewer questions than <<per_section>>. This is only a safeguard, not a target – if the content allows more, create more.
- Exception: only if the text of a section really contains no subject matter, return for it a single object {"lo_id": <id>, "reason": "<short reason>"} instead of questions.

CONTENT:
- Questions are based only on the text of their section. Do NOT invent information that is not in the text.
- The question is a standalone, content-focused question about the subject matter (definition, difference, cause, procedure, property, calculation, application).
- The student has neither the text nor the document. Do NOT write "Read the text...", "Based on the text..." and do not refer to pages, chapters, figures or parts of the document (not "according to the example on page 66"). If a question builds on an example from the text, give all its data in the question.
- Write the question, choices and keywords in English.

QUESTION TYPES:
<<type_rules>>

RULES PER TYPE:
- "choice": exactly <<choice_options>> options in "choices". One or more options are correct ("is_correct": true), the others have "is_correct": false. Wrong options are plausible and come from the same subject matter (confused concepts, a common mistake, a wrong calculation). Do not use options like "all of the above" or "none of the above". No two options have the same text or meaning. Vary the position of the correct options.
- "open": exactly one entry in "choices": {"text": "<model answer>", "is_correct": true}. The model answer is short and exact (a term, a number with its unit or one sentence), because the teacher grades it by hand or by exact match. For a practical task ("practical": true) it is a model solution with the steps and the result.
- "no_answer": a task that is not graded (write an essay, reflect, discuss). The task is clear and related to the subject matter of the section. "choices" is an empty array [].

TIMES:
- "time_to_handle" is the time in seconds the student needs to work on the question; "time_to_explain" is the time the teacher needs to explain the solution. Estimate them from the difficulty of the question.
- Ranges (time_to_handle / time_to_explain): choice 30–180 / 0–90, open 120–900 / 0–180, no_answer 0–1800 / 0–600.

Sections:
<<lo_blocks>>
<<extra_section>>
Output format:
Return ONLY a valid JSON object {"items": [...]} with no extra text. Every entry of "items" has the fields:
- "lo_id": id of the section the question belongs to,
- "name": short name of the question (at most 60 characters),
- "description": text of the question or task,
- "type": "choice", "open" or "no_answer",
- "practical": true for a practical task (calculation, procedure), else false,
- "choices": array of objects {"text": "...", "is_correct": true or false} by the rules per type,
- "time_to_handle": integer (seconds),
- "time_to_explain": integer (seconds),
- "bloom_level": level of the question, one of these exact Slovak values: Zapamätať si, Pochopiť, Aplikovať, Analyzovať, Hodnotiť, Vytvoriť,
- "keywords": 1 to 5 keywords of the question,
- "source": pages the question is based on, as document:page, e.g. ["D1:12", "D1:13"].
"""

EXTRA_ROUND_SK = """
DOPLŇUJÚCE KOLO:
Pre tieto tematické celky už existujú otázky uvedené nižšie. Vytvor ďalšie otázky, ktoré sa od nich líšia obsahom alebo uhlom pohľadu: iný pojem, vzorec, postup, príklad alebo súvislosť, iná úroveň Bloomovej taxonómie alebo iný typ. Existujúce otázky nepreformulúvaj. Ak pre tematický celok už nič nové nie je, nevracaj preň nič. Ak nič nové nie je pre žiadny celok, vráť {"items": []}. Spodná hranica počtu otázok v tomto kole neplatí.

Už vytvorené otázky:
<<existing>>
"""

EXTRA_ROUND_EN = """
ADDITIONAL ROUND:
These sections already have the questions listed below. Create more questions that differ from them in content or angle: another concept, formula, procedure, example or relationship, another level of Bloom's taxonomy or another type. Do not rephrase the existing questions. If there is nothing new for a section, return nothing for it. If there is nothing new for any section, return {"items": []}. The lower bound on the number of questions does not apply in this round.

Existing questions:
<<existing>>
"""

COVERAGE_SK = """
DOPLNENIE POKRYTIA:
Tieto tematické celky majú po kontrole kvality menej otázok, ako je potrebné. Pre každý celok vytvor aspoň toľko nových otázok, koľko je uvedené pri celku („chýba“). Ak to obsah dovoľuje, vytvor aj viac, lebo aj nové otázky prejdú kontrolou.

Najdôležitejšie je, aby otázka bola vecne správna a aby sa jej správna odpoveď dala jednoznačne zistiť iba z textu tematického celku. Radšej jednoduchšia otázka na pojem, definíciu, vlastnosť alebo postup, ktorý text výslovne uvádza, ako náročná otázka, ktorej odpoveď text priamo neobsahuje.

Pri každej už vytvorenej otázke je uvedené, či bola prijatá alebo vyradená a prečo:
- „nezodpovedateľná zo zdroja“: odpoveď sa z textu celku nedá jednoznačne zistiť – pýtaj sa iba na to, čo text uvádza,
- „nevychádza z textu“: otázka alebo odpoveď obsahovala informácie, ktoré v texte nie sú,
- „zle označená možnosť“: označenie správnej a nesprávnych možností nezodpovedalo textu – každú možnosť over oproti textu,
- „nízke skóre kvality“: zadanie bolo nejasné alebo viacznačné, prípadne odpoveď nebola správna,
- „duplicita“: otázka bola príliš podobná inej otázke celku,
- „neplatný formát“ alebo „nepovolený typ“: otázka nedodržala formát výstupu alebo pravidlá typov.
Vyhni sa chybám vyradených otázok. Vyradené ani prijaté otázky neopakuj a nepreformulúvaj.

Už vytvorené otázky:
<<existing>>
"""

COVERAGE_EN = """
COVERAGE SUPPLEMENT:
These sections have fewer questions than needed after the quality check. For each section, create at least as many new questions as listed next to it ("missing"). If the content allows, create more, because the new questions also go through the check.

Most important: each question must be factually correct, and its correct answer must be clearly found in the text of its section alone. Prefer a simpler question about a concept, definition, property or procedure that the text states explicitly over a harder question whose answer the text does not contain directly.

Each existing question is marked as accepted or dropped, with the reason:
- "not answerable from the source": the answer cannot be clearly found in the text of the section – ask only about what the text states,
- "not supported by the text": the question or answer contained information that is not in the text,
- "wrongly marked option": the marking of the correct and wrong options did not match the text – check every option against the text,
- "low quality score": the question was unclear or ambiguous, or the answer was not correct,
- "duplicate": the question was too similar to another question of the section,
- "invalid format" or "type not allowed": the question did not follow the output format or the type rules.
Avoid the mistakes of the dropped questions. Do not repeat or rephrase dropped or accepted questions.

Existing questions:
<<existing>>
"""

# Status marks of the existing questions in the coverage prompt. Keys: "accepted", the drop
# reasons of the pipeline, and for low_score the failed checks (quality, faithfulness,
# answerability). Unknown keys are shown as dropped without a reason.
STATUS_LABELS = {
    "sk": {
        "accepted": "prijatá", "dropped": "VYRADENÁ",
        "quality": "nízke skóre kvality", "faithfulness": "nevychádza z textu",
        "answerability": "nezodpovedateľná zo zdroja", "wrong_choice_marking": "zle označená možnosť",
        "duplicate": "duplicita", "validation": "neplatný formát",
        "type_not_allowed": "nepovolený typ", "type_rule": "nepovolený typ",
        "accepted_count": "prijaté",
    },
    "en": {
        "accepted": "accepted", "dropped": "DROPPED",
        "quality": "low quality score", "faithfulness": "not supported by the text",
        "answerability": "not answerable from the source", "wrong_choice_marking": "wrongly marked option",
        "duplicate": "duplicate", "validation": "invalid format",
        "type_not_allowed": "type not allowed", "type_rule": "type not allowed",
        "accepted_count": "accepted",
    },
}

# First lines of the generation prompts (qgen.llm.adapters.replay.QUESTION_PROMPT_MARKERS).
GENERATION_MARKERS = (
    "Z tematických celkov učebného materiálu a ich textu vytvor otázky a úlohy pre študentov.",
    "From the sections of the study material and their text, create questions and tasks for students.",
)


def build_lo_blocks(blocks: list[dict], document_language: str = "sk") -> str:
    """blocks: {"id", "name", "bloom", "context"} per section."""
    title = "Section" if document_language == "en" else "Tematický celok"
    parts = []
    for i, block in enumerate(blocks, start=1):
        parts.append(
            f"{title} {i}:\n"
            f"- lo_id: {block['id']}\n"
            f"- nazov: {block['name']}\n"
            f"- bloom_level: {block['bloom']}\n"
            f"- text:\n\"\"\"{block['context']}\"\"\""
        )
    return "\n\n".join(parts)


def _existing_lines(existing: dict, document_language: str, missing: dict | None = None,
                    statuses: dict | None = None) -> str:
    """statuses (coverage prompt only): question id -> list of status keys of STATUS_LABELS
    (["accepted"], ["duplicate"], ["quality", "answerability"], ...)."""
    english = document_language == "en"
    label = "missing" if english else "chýba"
    marks = STATUS_LABELS["en" if english else "sk"]
    lines = []
    for lo_id, questions in existing.items():
        head = f"- lo_id {lo_id}"
        if missing is not None:
            extra = ""
            if statuses is not None:
                accepted = sum(1 for q in questions if statuses.get(q.get("id")) == ["accepted"])
                extra = f", {marks['accepted_count']}: {accepted}"
            head += f" ({label}: {missing.get(lo_id, 1)}{extra})"
        lines.append(head + ":")
        for n, question in enumerate(questions, start=1):
            text = " ".join(str(question.get("description", "")).split())
            line = f"  {n}. [{question.get('type', '')}] {text}"
            if statuses is not None:
                line += " — " + _status_mark(statuses.get(question.get("id")) or [], marks)
            lines.append(line)
    return "\n".join(lines)


def _status_mark(keys: list[str], marks: dict) -> str:
    if keys == ["accepted"]:
        return marks["accepted"]
    reasons = []
    for key in keys:
        text = marks.get(key)
        if text and text not in reasons and key not in ("accepted", "dropped", "accepted_count"):
            reasons.append(text)
    return f"{marks['dropped']}: {', '.join(reasons)}" if reasons else marks["dropped"]


def build_extra_round_section(existing: dict, document_language: str = "sk") -> str:
    """existing: lo_id -> questions already created (only type and description are listed)."""
    template = EXTRA_ROUND_EN if document_language == "en" else EXTRA_ROUND_SK
    return _fill(template, existing=_existing_lines(existing, document_language))


def build_coverage_section(existing: dict, missing: dict, document_language: str = "sk",
                           statuses: dict | None = None) -> str:
    """missing: lo_id -> number of questions still needed to reach per_section; statuses:
    question id -> status keys (accepted or why it was dropped, see STATUS_LABELS)."""
    template = COVERAGE_EN if document_language == "en" else COVERAGE_SK
    return _fill(template, existing=_existing_lines(existing, document_language, missing, statuses or {}))


def build_item_generation_prompt(
    lo_blocks: str,
    document_language: str = "sk",
    qs: QuestionSettings | None = None,
    settings: Settings | None = None,
    extra_section: str = "",
) -> str:
    qs = qs or QuestionSettings()
    settings = settings or Settings.model_construct()
    english = document_language == "en"
    return _fill(
        GENERATION_EN if english else GENERATION_SK,
        per_section=qs.per_section,
        type_rules=type_rules_text(qs, settings, document_language),
        choice_options=qs.choice_options,
        lo_blocks=lo_blocks,
        extra_section=extra_section,
    )


# --- record texts (quality score, faithfulness, answerability) -------------------------------

_LABELS = {
    "sk": {
        "type": "typ", "question": "zadanie", "choices": "možnosti", "correct": "správna", "wrong": "nesprávna",
        "answer": "vzorová odpoveď", "no_answer": "odpoveď: (úloha sa nevyhodnocuje)",
    },
    "en": {
        "type": "type", "question": "question", "choices": "options", "correct": "correct", "wrong": "wrong",
        "answer": "model answer", "no_answer": "answer: (the task is not graded)",
    },
}


def _labels(document_language: str) -> dict:
    return _LABELS["en" if document_language == "en" else "sk"]


def question_record_text(item: dict, document_language: str = "sk", *, marks: bool, prefix: str = "") -> str:
    """Type, question and answer of one question as prompt lines.

    marks=True: choice options with [correct]/[wrong] and the model answer of open
    (quality score, faithfulness); marks=False: options without marks and no answer
    (answerability).
    """
    labels = _labels(document_language)
    question_type = item.get("type", "")
    lines = [
        f"{prefix}{labels['type']}: {question_type}",
        f"{prefix}{labels['question']}: {item.get('description', '')}",
    ]
    choices = [c for c in item.get("choices") or [] if isinstance(c, dict)]
    if question_type == "choice":
        lines.append(f"{prefix}{labels['choices']}:")
        for letter, choice in zip(LETTERS, choices):
            mark = f" [{labels['correct'] if choice.get('is_correct') is True else labels['wrong']}]" if marks else ""
            lines.append(f"  {letter}) {choice.get('text', '')}{mark}")
    elif question_type == "open":
        if marks:
            answer = choices[0].get("text", "") if choices else ""
            lines.append(f"{prefix}{labels['answer']}: {answer}")
    elif marks:
        lines.append(f"{prefix}{labels['no_answer']}")
    return "\n".join(lines)


EVALUATION_SK = """
Si učiteľ, ktorý hodnotí kvalitu otázok a úloh pre študentov.
Pre každú otázku priraď skóre od 1 do 5 (1 = veľmi slabé, 5 = výborné) a krátke zdôvodnenie (1–2 vety).

Hodnoť podľa:
- vecnej správnosti zadania a odpovede,
- jasnosti a jednoznačnosti zadania,
- pri "choice": či sú nesprávne možnosti vierohodné a z toho istého učiva a či otázka nemá viac výkladov,
- pri "open": či je vzorová odpoveď krátka a presná; pri praktickej úlohe či je riešenie správne a má postup,
- pri "no_answer": či je zadanie jasné a súvisí s učivom.

Otázky na hodnotenie:
<<items_text>>

Vráť LEN validný JSON objekt bez ďalšieho textu:
{"items": [{"id": 123, "skore": 4, "zdovodnenie": "stručné zdôvodnenie"}]}
"""

EVALUATION_EN = """
You are a teacher evaluating the quality of questions and tasks for students.
For each question assign a score from 1 to 5 (1 = very weak, 5 = excellent) and a short justification in English (1-2 sentences).

Evaluate according to:
- factual correctness of the question and the answer,
- clarity and unambiguity of the question,
- for "choice": whether the wrong options are plausible and from the same subject matter, and whether the question has only one reading,
- for "open": whether the model answer is short and exact; for a practical task whether the solution is correct and shows the steps,
- for "no_answer": whether the task is clear and related to the subject matter.

Questions to evaluate:
<<items_text>>

Return ONLY a valid JSON object with no extra text:
{"items": [{"id": 123, "skore": 4, "zdovodnenie": "short justification"}]}
"""


def evaluation_items_text(items: list[dict], document_language: str = "sk") -> str:
    parts = []
    for item in items:
        parts.append(
            f"Item id: {item.get('id')}\n"
            f"- lo_id: {item.get('lo_id')}\n"
            + question_record_text(item, document_language, marks=True, prefix="- ")
        )
    return "\n\n".join(parts)


def build_item_evaluation_prompt(items_text: str, document_language: str = "sk") -> str:
    return _fill(EVALUATION_EN if document_language == "en" else EVALUATION_SK, items_text=items_text)
