"""Question types choice / open / no_answer (kap. 7): settings, normalization, rules.

The question pipeline returns questions in the internal format below. The first part has
exactly the fields of the contract `Question` (qgen.contract.units); the rest is metadata
for the report (kap. 6, question_meta):

    id, lo_id
    name, description, type, time_to_handle, time_to_explain, hints, choices   (contract)
    bloom_level   remember | understand | apply | analyze | evaluate | create
                  (from the LLM; missing or unknown -> the section's level)
    keywords      1-5 strings (missing -> [section title])
    source        cited pages, e.g. ["D1:4"]          (former citovane_zdroje)
    source_files  names of the cited PDFs              (former zdroj)
    practical     practical task (calculation, procedure)
    origin        initial | extra_<n> | coverage
    evaluation    {"score": 1-5 | None, "reason": str} (quality score, former hodnotenie)

Length limits of the contract (255 / 4000 / 50 / 2000) are not checked here; that is the
output step (krok 14).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Optional

from qgen.config import Settings
from qgen.contract.options import OptionDefaults, QuestionOptions
from qgen.contract.units import Question

QUESTION_TYPES: tuple[str, ...] = ("choice", "open", "no_answer")
CONTRACT_FIELDS: tuple[str, ...] = tuple(Question.model_fields)

# Times per type in seconds (kap. 7): allowed ranges and defaults (time_to_handle, time_to_explain).
TIME_RANGES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "choice": ((30, 180), (0, 90)),
    "open": ((120, 900), (0, 180)),
    "no_answer": ((0, 1800), (0, 600)),
}
DEFAULT_TIMES: dict[str, tuple[int, int]] = {"choice": (120, 30), "open": (300, 0), "no_answer": (300, 0)}

BLOOM_LEVELS: dict[str, str] = {
    "zapamätať si": "remember",
    "zapamätať": "remember",
    "pochopiť": "understand",
    "porozumieť": "understand",
    "aplikovať": "apply",
    "analyzovať": "analyze",
    "hodnotiť": "evaluate",
    "vytvoriť": "create",
    "remember": "remember",
    "understand": "understand",
    "apply": "apply",
    "analyze": "analyze",
    "evaluate": "evaluate",
    "create": "create",
}
BLOOM_FALLBACK = "understand"
# Slovak labels used in the prompts (the LO pipeline uses them for both languages).
BLOOM_LABELS: dict[str, str] = {
    "remember": "Zapamätať si",
    "understand": "Pochopiť",
    "apply": "Aplikovať",
    "analyze": "Analyzovať",
    "evaluate": "Hodnotiť",
    "create": "Vytvoriť",
}

HINT_NAMES: dict[str, dict[int, tuple[str, ...]]] = {
    "sk": {1: ("Nápoveda",), 2: ("Ľahká nápoveda", "Ťažká nápoveda")},
    "en": {1: ("Hint",), 2: ("Easy hint", "Hard hint")},
}

# A hint (or question) that points at the document; the student has no PDF in ECEH. Warning only.
PAGE_REFERENCE_RE = re.compile(r"stran|strana|kapitol", re.IGNORECASE)
# Choices that are not allowed in a choice question (error).
FORBIDDEN_CHOICE_RE = re.compile(
    r"\b(všetky|žiadna|žiadne|ani jedna)\s+(z\s+)?(uveden|vyššie|horeuveden)|"
    r"\b(all|none)\s+of\s+the\s+(above|options|answers)\b",
    re.IGNORECASE,
)

ORIGIN_INITIAL = "initial"
ORIGIN_COVERAGE = "coverage"


def extra_origin(round_no: int) -> str:
    return f"extra_{round_no}"


@dataclass(frozen=True)
class QuestionSettings:
    """options.questions of the job (Príloha A) with the defaults filled in."""

    types: tuple[str, ...] = OptionDefaults.question_types
    per_section: int = OptionDefaults.per_section
    choice_options: int = OptionDefaults.choice_options
    hints: int = OptionDefaults.hints

    @classmethod
    def from_options(
        cls, options: Optional[QuestionOptions] = None, per_section: Optional[int] = None
    ) -> "QuestionSettings":
        defaults = OptionDefaults()
        options = options or QuestionOptions()
        types = tuple(t for t in QUESTION_TYPES if t in (options.types or defaults.question_types))
        return cls(
            types=types or defaults.question_types,
            per_section=per_section if per_section is not None else (options.per_section or defaults.per_section),
            choice_options=options.choice_options or defaults.choice_options,
            hints=options.hints if options.hints is not None else defaults.hints,
        )


# --- small helpers ------------------------------------------------------------------------


def bloom_code(value: Any) -> Optional[str]:
    return BLOOM_LEVELS.get(str(value or "").strip().casefold())


def section_bloom(section: Optional[dict]) -> str:
    return bloom_code((section or {}).get("bloom_level")) or BLOOM_FALLBACK


def bloom_labels(codes) -> list[str]:
    """Slovak labels of Bloom codes, in the given order (unknown codes left out)."""
    labels = [BLOOM_LABELS[c] for c in codes if c in BLOOM_LABELS]
    return labels


def hint_names(count: int, language: str) -> tuple[str, ...]:
    names = HINT_NAMES.get(language, HINT_NAMES["sk"])
    if count <= 0:
        return ()
    if count in names:
        return names[count]
    base = names[2]
    return tuple(base[i] if i < len(base) else f"{base[-1]} {i + 1}" for i in range(count))


def clamp_times(question_type: str, handle: Any, explain: Any) -> tuple[int, int, bool]:
    """(time_to_handle, time_to_explain, changed): default when missing, clamped to the range."""
    default_handle, default_explain = DEFAULT_TIMES.get(question_type, DEFAULT_TIMES["open"])
    (h_min, h_max), (e_min, e_max) = TIME_RANGES.get(question_type, TIME_RANGES["open"])
    changed = False
    values = []
    for raw, default, low, high in ((handle, default_handle, h_min, h_max), (explain, default_explain, e_min, e_max)):
        value = _as_int(raw)
        if value is None:
            value = default
        clamped = max(low, min(high, value))
        if clamped != value:
            changed = True
        values.append(clamped)
    return values[0], values[1], changed


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return None
    return None


def hint_html(text: str) -> str:
    """HTML of a hint: kept when it already starts with a tag, else escaped into <p>…</p>.

    Cleaning with nh3 (allowed tags only) is done by the output step.
    """
    text = text.strip()
    if text.startswith("<"):
        return text
    return f"<p>{html.escape(text, quote=False)}</p>"


def contract_part(item: dict) -> dict:
    """The fields of the contract Question, in contract order."""
    return {name: item.get(name) for name in CONTRACT_FIELDS}


def correct_texts(item: dict) -> list[str]:
    return [str(c.get("text", "")) for c in item.get("choices") or [] if isinstance(c, dict) and c.get("is_correct") is True]


def normalize_sources(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    return [text] if text else []


# --- normalization --------------------------------------------------------------------------


def normalize_generated_item(
    raw: dict, section: Optional[dict], qs: QuestionSettings, language: str = "sk"
) -> tuple[dict, dict]:
    """One LLM item -> internal format. Returns (item, notes); notes: times_clamped (bool).

    Normalization never drops an item; rule violations are left for validate_items (e.g.
    is_correct that is not a bool stays as it is and fails the validation).
    """
    question_type = str(raw.get("type", "") or "").strip().lower()
    choices = []
    raw_choices = raw.get("choices") if isinstance(raw.get("choices"), list) else []
    for choice in raw_choices:
        if isinstance(choice, dict):
            choices.append({"text": str(choice.get("text", "") or "").strip(), "is_correct": choice.get("is_correct")})
        else:
            choices.append({"text": str(choice).strip(), "is_correct": None})

    raw_hints = raw.get("hints")
    if isinstance(raw_hints, str):
        raw_hints = [raw_hints]
    hint_texts = [str(h).strip() for h in (raw_hints if isinstance(raw_hints, list) else []) if str(h or "").strip()]
    hint_texts = hint_texts[: qs.hints]
    names = hint_names(len(hint_texts), language)
    hints = [{"name": name, "url": hint_html(text)} for name, text in zip(names, hint_texts)]

    handle, explain, times_clamped = clamp_times(question_type, raw.get("time_to_handle"), raw.get("time_to_explain"))

    bloom = bloom_code(raw.get("bloom_level")) or section_bloom(section)
    raw_keywords = raw.get("keywords")
    if isinstance(raw_keywords, str):
        raw_keywords = [k for k in raw_keywords.split(",")]
    keywords = [str(k).strip() for k in (raw_keywords if isinstance(raw_keywords, list) else []) if str(k or "").strip()]
    keywords = list(dict.fromkeys(keywords))[:5]
    if not keywords:
        title = str((section or {}).get("vzdelávací_objekt", "") or "").strip()
        keywords = [title] if title else []

    item = {
        "lo_id": raw.get("lo_id"),
        "name": str(raw.get("name", "") or "").strip(),
        "description": str(raw.get("description", "") or "").strip(),
        "type": question_type,
        "time_to_handle": handle,
        "time_to_explain": explain,
        "hints": hints,
        "choices": choices,
        "bloom_level": bloom,
        "keywords": keywords,
        "source": normalize_sources(raw.get("source", [])),
        "practical": raw.get("practical") is True,
    }
    if not item["name"]:
        item["name"] = str((section or {}).get("vzdelávací_objekt", "") or "").strip() or item["description"][:60]
    return item, {"times_clamped": times_clamped}


# --- type rules -----------------------------------------------------------------------------


def type_permission_problem(item: dict, qs: QuestionSettings, settings: Settings) -> Optional[str]:
    """'type_not_allowed' / 'type_rule' when the type breaks the options or the Bloom rule."""
    question_type = item.get("type")
    if question_type in QUESTION_TYPES and question_type not in qs.types:
        return "type_not_allowed"
    if question_type == "no_answer" and qs.types != ("no_answer",):
        if item.get("bloom_level") not in settings.no_answer_bloom_levels:
            return "type_rule"
    return None


def _join(labels: list[str], language: str) -> str:
    word = " alebo " if language == "sk" else " or "
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + word + labels[-1]


def _quoted(types, language: str) -> str:
    quoted = [f'"{t}"' for t in types]
    word = " a " if language == "sk" else " and "
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + word + quoted[-1]


def type_rules_text(qs: QuestionSettings, settings: Settings, language: str = "sk") -> str:
    """Block {type_rules} of the generation prompt, built from the options and the config."""
    types = qs.types
    share = int(round(settings.question_choice_min_share * 100))
    practical = _join(bloom_labels(settings.practical_bloom_levels), language)
    no_answer = _join(bloom_labels(settings.no_answer_bloom_levels), language)
    sk = language != "en"

    if types == ("choice",):
        return (
            '- Povolený je iba typ "choice". Všetky otázky sú "choice", aj výpočty a postupy (výsledky sú možnosti).'
            if sk else
            '- The only allowed type is "choice". All questions are "choice", also calculations and procedures '
            "(the results are the options)."
        )
    if types == ("no_answer",):
        return (
            '- Povolený je iba typ "no_answer". Všetky otázky sú úlohy bez vyhodnotenia.'
            if sk else
            '- The only allowed type is "no_answer". All questions are tasks that are not graded.'
        )

    lines = [
        f"- Povolené typy sú iba {_quoted(types, language)}." if sk else
        f"- The only allowed types are {_quoted(types, language)}."
    ]
    if "choice" in types:
        lines.append(
            f'- Prednostný typ je "choice": aspoň {share} % otázok každého tematického celku je "choice".' if sk else
            f'- The preferred type is "choice": at least {share} % of the questions of each section are "choice".'
        )
    elif "open" in types:
        lines.append('- Prednostný typ je "open".' if sk else '- The preferred type is "open".')
    if "open" in types:
        lines.append(
            '- "open" použi tam, kde sa odpoveď nedá dobre vybrať z možností (vysvetlenie vlastnými slovami, '
            "odvodenie, výpočet s postupom)." if sk else
            '- Use "open" where the answer cannot be picked well from options (explanation in own words, '
            "derivation, calculation with steps)."
        )
        if practical:
            lines.append(
                f"- Tematický celok s bloom_level {practical} má aj aspoň jednu praktickú úlohu: typ \"open\", "
                '"practical": true (výpočet alebo postup na konkrétnych údajoch).' if sk else
                f"- A section with bloom_level {practical} also gets at least one practical task: type \"open\", "
                '"practical": true (a calculation or procedure on concrete data).'
            )
    elif practical:
        lines.append(
            '- Praktické úlohy (výpočet, postup) zadaj ako "choice" s výsledkami ako možnosťami a "practical": true.'
            if sk else
            '- Give practical tasks (calculation, procedure) as "choice" with the results as options and "practical": true.'
        )
    if "no_answer" in types:
        lines.append(
            f'- "no_answer" použi iba pri otázke s bloom_level {no_answer}.' if sk else
            f'- Use "no_answer" only for a question with bloom_level {no_answer}.'
        )
    else:
        lines.append('- Typ "no_answer" nepoužívaj.' if sk else '- Do not use the type "no_answer".')
    return "\n".join(lines)


def hints_rule_text(qs: QuestionSettings, language: str = "sk") -> str:
    sk = language != "en"
    if qs.hints <= 0:
        return '"hints" je prázdne pole [].' if sk else '"hints" is an empty array [].'
    if qs.hints == 1:
        return 'Každá otázka má v "hints" presne 1 nápovedu.' if sk else 'Every question has exactly 1 hint in "hints".'
    if qs.hints == 2:
        return (
            'Každá otázka má v "hints" presne 2 nápovedy: prvá ľahká (všeobecné nasmerovanie), druhá ťažká '
            "(konkrétnejšia stopa)." if sk else
            'Every question has exactly 2 hints in "hints": the first easy (general direction), the second hard '
            "(a more specific clue)."
        )
    return (
        f'Každá otázka má v "hints" presne {qs.hints} nápovedy od najľahšej po najťažšiu.' if sk else
        f'Every question has exactly {qs.hints} hints in "hints", from the easiest to the hardest.'
    )
