import time
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from qgen.config import Settings, get_settings, setting_or_default
from qgen.context_builder import (
    build_allowed_source_refs,
    build_page_map,
    build_context_for_lo,
    build_source_name_map,
    parse_source_refs,
    resolve_source_names,
)
from qgen.extraction.document_language import resolve_document_language
from qgen.questions.answerability import analyze_item_answerability
from qgen.questions.faithfulness import analyze_item_faithfulness
from qgen.questions.relevance_to_lo import analyze_item_relevance_to_lo
from qgen.questions.validation import validate_items
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.llm.retry import generate_json
from qgen.progress import (
    STAGE_Q_EVALUATION,
    STAGE_QUESTIONS,
    ProgressCallback,
    SubProgress,
    batch_count,
    report_progress,
)


@dataclass
class QuestionPipelineResult:
    """Result of generate_all_items. Nothing is written to disk.

    questions      accepted questions with ids 1..n
    all_questions  all questions before filtering
    language_info  {"language": "sk"|"en", "reason": str}
    reports        report dicts, keys: item_validation, item_relevance_to_lo,
                   item_faithfulness, item_answerability, item_processing_time
    """

    questions: list[dict]
    all_questions: list[dict]
    language_info: dict[str, str]
    reports: dict[str, Any] = field(default_factory=dict)


def generate_items_for_batch(
    los_batch,
    page_map,
    document_language="sk",
    model=None,
    llm=None,
    verbose=True,
    context_max_chars=None,
):
    model = setting_or_default(model, "llm_model_questions")
    lo_blocks = []
    for lo in los_batch:
        lo_id = lo.get("id")
        context = build_context_for_lo(lo, page_map, max_chars=context_max_chars)
        if not context.strip():
            if verbose:
                print(f"  LO {lo_id}: bez kontextu, preskakujem v tomto batche.")
            continue
        lo_blocks.append({
            "id": lo_id,
            "name": lo.get("vzdelávací_objekt", ""),
            "bloom": lo.get("bloom_level", ""),
            "context": context,
        })

    if not lo_blocks:
        return []

    parts = []
    for i, block in enumerate(lo_blocks, start=1):
        parts.append(
            f"LO {i}:\n"
            f"- lo_id: {block['id']}\n"
            f"- nazov: {block['name']}\n"
            f"- bloom_level: {block['bloom']}\n"
            f"- text:\n\"\"\"{block['context']}\"\"\""
        )
    los_text = "\n\n".join(parts)

    prompt = build_item_generation_prompt(
        los_text,
        document_language=document_language,
    )

    # Step 6: no `except Exception: return []` (the batch was retried and then left without
    # questions); LLMError propagates.
    parsed = generate_json(llm, prompt, role="questions", model=model, verbose=verbose)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "lo_id" not in item:
            continue
        items.append(normalize_generated_item(item))
    return items


def evaluate_items_batch(items_batch, model=None, llm=None, verbose=True, document_language="sk"):
    if not items_batch:
        return {}
    model = setting_or_default(model, "llm_model_eval")

    parts = []
    for item in items_batch:
        parts.append(
            f"Item id: {item.get('id')}\n"
            f"- lo_id: {item.get('lo_id')}\n"
            f"- typ: {item.get('typ', '')}\n"
            f"- otazka: {item.get('otazka', '')}\n"
            f"- odpoved: {item.get('odpoved', '')}\n"
            f"- napoveda: {item.get('napoveda', '')}\n"
            f"- citovane_zdroje: {item.get('citovane_zdroje', [])}"
        )
    items_text = "\n\n".join(parts)

    prompt = build_item_evaluation_prompt(items_text, document_language)

    # Step 6: no `except Exception: return {}` (the questions then had no score and were
    # filtered out); LLMError propagates.
    parsed = generate_json(llm, prompt, role="eval", model=model, verbose=verbose)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return {}

    evaluations = {}
    for raw in parsed:
        if not isinstance(raw, dict):
            continue
        item_id = raw.get("id")
        if item_id is None:
            continue
        try:
            score = int(raw.get("skore"))
        except (TypeError, ValueError):
            continue
        score = max(1, min(5, score))
        reason = str(raw.get("zdovodnenie", "")).strip()
        evaluations[item_id] = {"skore": score, "zdovodnenie": reason}
    return evaluations


def item_sort_key(item, lo_order):
    lo_id = item.get("lo_id")
    lo_pos = lo_order.get(lo_id, float("inf"))
    refs = parse_source_refs(item.get("citovane_zdroje", []))
    source_id, first_page = refs[0] if refs else ("", float("inf"))
    return (lo_pos, source_id or "", first_page, item.get("id", float("inf")))


def attach_source_names(items, source_name_map):
    for item in items:
        item["zdroj"] = resolve_source_names(item.get("citovane_zdroje", []), source_name_map)
    return items


def generate_all_items(
    los,
    segmenty,
    batch_size=None,
    evaluation_batch_size=None,
    model=None,
    generation_model=None,
    evaluation_model=None,
    llm: Optional[LLMClient] = None,
    embedder: Optional[EmbeddingClient] = None,
    verbose=True,
    max_batch_attempts=None,
    max_eval_attempts=None,
    document_language=None,
    progress_callback: Optional[ProgressCallback] = None,
    settings: Optional[Settings] = None,
) -> QuestionPipelineResult:
    """Question pipeline: language, generation and scoring per LO batch, evaluation, filter.

    document_language: None or "auto" = detect with the LLM; "sk" / "en" = use it.
    Values not given (None) come from `settings` (default: get_settings()).
    llm / embedder = None: created by qgen.llm.factory per role.
    An LLM or embedding call that fails for good raises LLMError.
    """
    settings = settings or get_settings()
    batch_size = setting_or_default(batch_size, "question_batch_size", settings)
    evaluation_batch_size = setting_or_default(evaluation_batch_size, "question_eval_batch_size", settings)
    max_batch_attempts = setting_or_default(max_batch_attempts, "question_generation_attempts", settings)
    max_eval_attempts = setting_or_default(max_eval_attempts, "question_eval_attempts", settings)
    generation_model = generation_model or model or settings.llm_model_questions
    evaluation_model = evaluation_model or model or settings.llm_model_eval

    page_map = build_page_map(segmenty)
    source_name_map = build_source_name_map(segmenty)
    if verbose:
        print("Item pipeline: detekujem jazyk dokumentu.")
    language_info = resolve_document_language(
        segmenty,
        document_language,
        llm=llm,
        model=generation_model,
        settings=settings,
        verbose=verbose,
        progress_callback=progress_callback,
        role="questions",
    )
    document_language = language_info.get("language", "sk")
    lo_order = {lo.get("id"): idx for idx, lo in enumerate(los)}
    all_items = []
    next_item_id = 1
    total_los = len(los)
    total_batches = batch_count(total_los, batch_size)
    report_progress(progress_callback, STAGE_QUESTIONS, 0, total_batches)

    if verbose:
        print(f"Začínam generovanie položiek pre {total_los} LO v batchoch po {batch_size}.")
    generation_seconds = 0.0
    evaluation_seconds = 0.0
    batch_num = 1
    for start in range(0, total_los, batch_size):
        batch = los[start:start + batch_size]
        lo_ids = [lo.get("id") for lo in batch]
        if verbose:
            print(f"\nBatch {batch_num}: LO id {lo_ids}")
        start_batch_generation = time.perf_counter()
        batch_generation_seconds = 0.0
        batch_evaluation_seconds = 0.0

        raw_items = []
        for attempt in range(1, max_batch_attempts + 1):
            raw_items = generate_items_for_batch(
                batch,
                page_map,
                document_language=document_language,
                model=generation_model,
                llm=llm,
                verbose=verbose,
                context_max_chars=settings.context_max_chars,
            )
            if raw_items:
                break
            if verbose and max_batch_attempts > 1:
                print(f"Batch {batch_num}: prazdny vystup, opakujem ({attempt}/{max_batch_attempts})")

        if not raw_items:
            if verbose:
                print(f"Batch {batch_num}: LLM nevrátil žiadne položky.")
        else:
            created_batch_items = []
            for raw in raw_items:
                lo_id = raw.get("lo_id")
                record = {
                    "id": next_item_id,
                    "lo_id": lo_id,
                    "typ": raw.get("typ", ""),
                    "otazka": raw.get("otazka", ""),
                    "odpoved": raw.get("odpoved", ""),
                    "napoveda": raw.get("napoveda", ""),
                    "citovane_zdroje": raw.get("citovane_zdroje", [])
                }
                record["zdroj"] = resolve_source_names(record.get("citovane_zdroje", []), source_name_map)
                created_batch_items.append(record)
                all_items.append(record)
                next_item_id += 1

            batch_generation_seconds = time.perf_counter() - start_batch_generation
            generation_seconds += batch_generation_seconds

            start_batch_evaluation = time.perf_counter()
            batch_eval = {}
            for attempt in range(1, max_eval_attempts + 1):
                batch_eval = evaluate_items_batch(
                    created_batch_items,
                    model=evaluation_model,
                    llm=llm,
                    verbose=verbose,
                    document_language=document_language,
                )
                if batch_eval:
                    break
                if verbose and max_eval_attempts > 1:
                    print(f"Batch {batch_num}: hodnotenie prazdne, opakujem ({attempt}/{max_eval_attempts})")

            for record in created_batch_items:
                item_eval = batch_eval.get(record["id"], {})
                record["hodnotenie"] = {
                    "skore": item_eval.get("skore"),
                    "zdovodnenie": item_eval.get("zdovodnenie", "")
                }
            batch_evaluation_seconds = time.perf_counter() - start_batch_evaluation
            evaluation_seconds += batch_evaluation_seconds

            if verbose:
                print(f"Batch {batch_num}: vytvorených položiek: {len(raw_items)}")
        if not raw_items:
            batch_generation_seconds = time.perf_counter() - start_batch_generation
            generation_seconds += batch_generation_seconds

        if verbose:
            print(
                f"Batch {batch_num} hotový. "
                f"Generovanie: {batch_generation_seconds:.2f} s, "
                f"evaluacia: {batch_evaluation_seconds:.2f} s"
            )
        report_progress(progress_callback, STAGE_QUESTIONS, batch_num, total_batches)
        batch_num += 1
    all_items.sort(key=lambda item: item_sort_key(item, lo_order))
    for i, item in enumerate(all_items, start=1):
        item["id"] = i
    attach_source_names(all_items, source_name_map)

    # Step q_evaluation: relevance (1 unit), faithfulness batches, answerability batches.
    expected_check_batches = batch_count(len(all_items), evaluation_batch_size)
    report_progress(progress_callback, STAGE_Q_EVALUATION, 0, 1 + 2 * expected_check_batches)
    evaluation_start_reports = time.perf_counter()
    if verbose:
        print("Item pipeline: spustam validaciu a evaluacne reporty poloziek.")
    allowed_pages = build_allowed_source_refs(segmenty)
    valid_lo_ids = {lo.get("id") for lo in los if isinstance(lo.get("id"), int)}
    item_validation_report = validate_items(
        all_items,
        allowed_pages=allowed_pages,
        valid_lo_ids=valid_lo_ids,
    )
    item_relevance_report = analyze_item_relevance_to_lo(
        all_items,
        los,
        embedder=embedder,
        embedding_model=settings.embed_model,
        verbose=verbose,
    )
    report_progress(progress_callback, STAGE_Q_EVALUATION, 1, 1 + 2 * expected_check_batches)
    faithfulness_progress = SubProgress(progress_callback, before=1, after=expected_check_batches)
    item_faithfulness_report = analyze_item_faithfulness(
        segmenty,
        all_items,
        llm=llm,
        model=settings.llm_model,
        verbose=verbose,
        batch_size=evaluation_batch_size,
        max_batch_attempts=settings.check_batch_attempts,
        document_language=document_language,
        context_max_chars=settings.context_max_chars,
        faithful_score=settings.question_faithful_score,
        progress_callback=faithfulness_progress,
    )
    answerability_progress = SubProgress(progress_callback, before=1 + faithfulness_progress.sub_total)
    item_answerability_report = analyze_item_answerability(
        segmenty,
        all_items,
        llm=llm,
        model=settings.llm_model,
        verbose=verbose,
        batch_size=evaluation_batch_size,
        max_batch_attempts=settings.check_batch_attempts,
        document_language=document_language,
        context_max_chars=settings.context_max_chars,
        answerable_score=settings.question_answerable_score,
        progress_callback=answerability_progress,
    )
    q_evaluation_done = 1 + faithfulness_progress.sub_total + answerability_progress.sub_total
    report_progress(progress_callback, STAGE_Q_EVALUATION, q_evaluation_done, q_evaluation_done)
    accepted_items = filter_items_variant_b(
        all_items,
        item_validation_report,
        item_faithfulness_report,
        item_answerability_report,
        min_score=settings.question_min_score,
        min_faithfulness_score=settings.question_min_faithfulness_score,
        min_answerability_score=settings.question_min_answerability_score,
    )
    normalized_items = normalize_accepted_items(accepted_items, valid_lo_ids)
    attach_source_names(normalized_items, source_name_map)
    evaluation_seconds += time.perf_counter() - evaluation_start_reports

    timing_report = {
        "pipeline": "items",
        "generation_seconds": round(generation_seconds, 4),
        "evaluation_seconds": round(evaluation_seconds, 4),
        "total_seconds": round(generation_seconds + evaluation_seconds, 4),
        "details": {
            "items_count_all": len(all_items),
            "items_count_accepted": len(normalized_items),
            "los_count": len(los),
            "document_language": document_language,
            "document_language_reason": language_info.get("reason", ""),
        },
    }

    if verbose:
        print(f"\nGenerovanie položiek pre všetky LO dokončené. Celkový počet položiek: {len(all_items)}")
        print(f"Cas generovania položiek: {generation_seconds:.2f} s")
        print(f"Cas evaluacie položiek: {evaluation_seconds:.2f} s")
    return QuestionPipelineResult(
        questions=normalized_items,
        all_questions=all_items,
        language_info=language_info,
        reports={
            "item_validation": item_validation_report,
            "item_relevance_to_lo": item_relevance_report,
            "item_faithfulness": item_faithfulness_report,
            "item_answerability": item_answerability_report,
            "item_processing_time": timing_report,
        },
    )


def build_item_generation_prompt(los_text, document_language):
    if document_language == "en":
        return f"""
You are an experienced teacher.
Based on multiple learning objectives (LO) and their textual context from the study material,
generate educational items (questions/tasks) for each LO.

REQUIREMENTS:
- For each LO, generate only items grounded in its text and focused on that learning objective.
- Do NOT invent information that is not present in the text.
- Every item must contain the field "lo_id".
- If the context does not support any reasonable item, create nothing for that LO.
- Write questions and tasks as standalone content-focused prompts, not as instructions to read or analyze the text.
- Do NOT generate meta-phrasing such as "Read the text..." or "Based on the text...".
- The question should target the subject matter directly.
- Keep JSON field names exactly as specified below.
- Because the source document is in English, write the content of the item, answer, and hint in English.
- Fill testing-related fields only for practical Python tasks.
- If the document is about C++, Java, JavaScript, C# or another language, the answer and code solution may use that language.
- Python testing fields belong ONLY to tasks solved in Python.
- If the solution is not in Python, set:
  "execution_mode": "",
  "function_name": "",
  "automaticky_testovatelna": false,
  "test_cases": []
- Set "automaticky_testovatelna" to true only when the task can be reliably checked automatically.
- GUI, interactive, graphical, web, or otherwise non-testable tasks must have "automaticky_testovatelna": false.
- If the task is not programming-related or not suitable for automatic testing, set:
  "jazyk": "",
  "kod_riesenia": "",
  "execution_mode": "",
  "function_name": "",
  "test_cases": [],
  "automaticky_testovatelna": false
- For practical Python tasks, provide runnable code in "kod_riesenia" and create 2 to 4 test cases.
- Allowed execution_mode values for Python practical tasks:
  "stdin_stdout" or "function"
- If you use "function", also fill "function_name".
- If you use "function", "kod_riesenia" may contain only function definitions plus optional imports or simple constants.
- In "function" mode, "kod_riesenia" must not contain prints, demo calls, fixed test inputs, or executable code outside the function definition.

Document classification:
- is_python_document: false
- reason: no justification

HARD RULE:
- If is_python_document = false, you must not generate Python practical tasks or Python tests.

Learning objectives:
{los_text}

Output format:
Return ONLY valid JSON as an array of objects.
Each object must have:
- "lo_id"
- "typ": "teoreticka_otazka" or "prakticka_uloha"
- "otazka"
- "odpoved"
- "napoveda"
- "citovane_zdroje": source refs like ["D1:12","D2:13"]
- "jazyk"
- "kod_riesenia"
- "execution_mode"
- "function_name"
- "automaticky_testovatelna"
- "test_cases"

Output:
ONLY a JSON array with no extra text.
"""
    return f"""
Si skusený učiteľ.
Na zaklade viacerých vzdelávacích cieľov (LO) a ich textového kontextu z učebného materiálu
vygeneruj vzdelávacie položky (otázky/úlohy) pre každý LO.

POŽIADAVKY:
- Pre každé LO generuj len položky, ktoré vychádzajú z jeho textu a sú tematicky zamerané na daný cieľ.
- NEvymýšľaj informácie, ktoré nie sú v texte.
- Pre každú položku vždy uveď, ku ktorému LO patrí, pomocou poľa "lo_id".
- Ak kontext neumožňuje žiadnu rozumnú položku, pre dané LO nevytváraj nič.
- Formuluj otázky a úlohy ako samostatné vecné zadania o téme, nie ako pokyny na prácu s textom.
- NEGENERUJ meta-formulácie typu "Prečítaj si text...", "Na základe textu..." ani otázky pýtajúce sa, čo študent pochopil z textu.
- Otázka má smerovať priamo na odborný obsah, napr. definíciu, rozdiel, príčinu, postup, vlastnosť, výpočet alebo aplikáciu.
- Testovateľné polia vypĺňaj len pre praktické úlohy v Pythone.
- Zohľadni jazyk zdrojového dokumentu.
- Ak je dokument o C++, Java, JavaScript, C# alebo inom jazyku, odpoveď a kód riešenia môžeš napísať v tomto jazyku.
- Do Python testovacej schémy patria LEN úlohy, ktorých riešenie je v jazyku Python, ak je dokument v inom jazyku tak testovacie polia nevyplnuj.
- Ak riešenie nie je v Pythone, nastav:
  "execution_mode": "",
  "function_name": "",
  "automaticky_testovatelna": false,
  "test_cases": []
- Pole "automaticky_testovatelna" nastav na true len vtedy, ak sa úloha dá spoľahlivo overiť automaticky cez vstup/výstup alebo volanie funkcie.
- Ak ide o GUI, interaktívnu, grafickú, webovú alebo inak netestovateľnú úlohu (napr. tkinter, pygame, turtle, streamlit), nastav "automaticky_testovatelna": false.
- Ak úloha nie je programovacia alebo nie je vhodná na automatické testovanie, nastav:
  "jazyk": "",
  "kod_riesenia": "",
  "execution_mode": "",
  "function_name": "",
  "test_cases": [],
  "automaticky_testovatelna": false
- Ak ide o praktickú úlohu s Python kódom, uveď spustiteľný kód v poli "kod_riesenia" a vytvor 2 až 4 test cases.
- Pre Python praktické úlohy používaj len tieto execution_mode:
  "stdin_stdout" alebo "function"
- Ak použiješ "function", doplň aj rovnake  "function_name" ktore sa ma použiť.
- Ak použiješ "function", "kod_riesenia" musí obsahovať len definície funkcií a prípadne importy alebo jednoduché konštanty.
- Pri "function" NESMIE "kod_riesenia" obsahovať printy, demo volania funkcie, pevne vložené testovacie vstupy ani spúšťací kód mimo definície funkcie.
- Pri "function" sa vstupy dodávajú len cez "test_cases".

Klasifikácia dokumentu:
- is_python_document: false
- reason: bez zdovodnenia

TVRDÉ PRAVIDLO:
- Ak is_python_document = false, NESMIEŠ generovať praktické úlohy s Python kódom ani Python testy.
- Ak is_python_document = false a vytvoríš praktickú úlohu, musí byť bez kódu a bez testovacích polí:
  "jazyk": "",
  "kod_riesenia": "",
  "execution_mode": "",
  "function_name": "",
  "automaticky_testovatelna": false,
  "test_cases": []

Zoznam LO:
{los_text}

Formát výstupu:
Vráť LEN validný JSON – pole objektov.
Každý objekt musí mať:
- "lo_id": id vzdelávacieho objektu, ku ktorému položka patrí,
- "typ": "teoreticka_otazka" alebo "prakticka_uloha",
- "otazka": zadanie otázky alebo úlohy,
- "odpoved": správna odpoveď alebo referenčné riešenie; ak ide o praktickú Python úlohu, pole MUSÍ obsahovať aj samotný kód riešenia, aby si ho používateľ vedel skopírovať,
- "napoveda": krátka pomocná stopa pre študenta, NESMIE obsahovať finálnu odpoveď ani kľúčový výsledok. Napoveda má len nasmerovať, čo si má študent v texte pozrieť alebo aký postup zvoliť.
- "citovane_zdroje": zoznam zdrojov vo formáte dokument:strana ako textové reťazce, napr. ["D1:12","D2:13"].
- "jazyk": napr. "python" alebo prázdny reťazec.
- "kod_riesenia": samostatný kód bez markdown ohraničenia alebo prázdny reťazec.
- "execution_mode": "stdin_stdout", "function" alebo prázdny reťazec.
- "function_name": názov funkcie pre execution_mode "function", inak prázdny reťazec.
- "automaticky_testovatelna": true alebo false.
- "test_cases": pole objektov; pre Python úlohy:
  - pri "stdin_stdout": {{"input": "...", "expected_output": "..."}}
  - pri "function": {{"input": [...], "expected_output": ...}}
  Inak prázdne pole [].
- Pre praktickú Python úlohu musia byť "odpoved" a "kod_riesenia" obsahovo konzistentné; "odpoved" má obsahovať kód alebo kód spolu s krátkym vysvetlením.

Výstup:
LEN JSON pole bez akéhokoľvek ďalšieho textu.
"""


def build_item_evaluation_prompt(items_text, document_language):
    if document_language == "en":
        return f"""
You are a teacher evaluating the quality of educational items.
For each item assign:
- a score from 1 to 5 (1 = very weak, 5 = excellent)
- a short justification in English (1-2 sentences)

Evaluate according to:
- factual correctness,
- clarity of the question/task wording,
- quality of the answer,
- quality of the hint.

Items to evaluate:
{items_text}

Return ONLY valid JSON as an array of objects:
[
  {{"id": 123, "skore": 4, "zdovodnenie": "short justification"}}
]

Output:
ONLY a JSON array with no extra text.
"""
    return f"""
Si učiteľ, ktorý hodnotí kvalitu vzdelávacích položiek.
Pre každú položku priraď:
- skore od 1 do 5 (1 = veľmi slabé, 5 = výborné)
- krátke zdôvodnenie (1-2 vety)

Hodnoť podľa:
- vecnej správnosti,
- jasnosti formulácie otázky/úlohy,
- kvality odpovede (ak je to prakticka úloha, hodnoti aj kód a či sa dá spustiť),
- vhodnosti nápovedy (nesmie prezrádzať finálne riešenie).

Položky na hodnotenie:
{items_text}

Vráť LEN validný JSON ako pole objektov:
[
  {{"id": 123, "skore": 4, "zdovodnenie": "stručné zdôvodnenie"}}
]

Výstup:
LEN JSON pole bez ďalšieho textu.
"""


def normalize_generated_item(item):
    normalized = {
        "lo_id": item.get("lo_id"),
        "typ": str(item.get("typ", "")).strip(),
        "otazka": str(item.get("otazka", "")).strip(),
        "odpoved": item.get("odpoved", ""),
        "napoveda": item.get("napoveda", ""),
        "citovane_zdroje": normalize_sources(item.get("citovane_zdroje", [])),
    }
    return normalized


def normalize_sources(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def filter_items_variant_b(
    items,
    validation_report,
    faithfulness_report,
    answerability_report,
    min_score=None,
    min_faithfulness_score=None,
    min_answerability_score=None,
):
    min_score = setting_or_default(min_score, "question_min_score")
    min_faithfulness_score = setting_or_default(min_faithfulness_score, "question_min_faithfulness_score")
    min_answerability_score = setting_or_default(min_answerability_score, "question_min_answerability_score")
    invalid_item_ids = extract_prefixed_ids(validation_report.get("errors", []), "Polozka")
    faithfulness_by_id = {
        row.get("item_id"): row.get("faithfulness_score")
        for row in faithfulness_report.get("items", [])
        if row.get("item_id") is not None
    }
    answerability_by_id = {
        row.get("item_id"): row.get("answerability_score")
        for row in answerability_report.get("items", [])
        if row.get("item_id") is not None
    }

    accepted = []
    for item in items:
        item_id = item.get("id")
        if item_id in invalid_item_ids:
            continue

        faithfulness_score = faithfulness_by_id.get(item_id)
        if faithfulness_score is None or faithfulness_score < min_faithfulness_score:
            continue

        score = get_item_score(item)
        answerability_score = answerability_by_id.get(item_id)
        score_ok = score is not None and score >= min_score
        answerability_ok = answerability_score is not None and answerability_score >= min_answerability_score
        if not (score_ok or answerability_ok):
            continue

        accepted.append(item)

    return accepted


def normalize_accepted_items(items, valid_lo_ids):
    normalized = []
    valid_lo_ids = set(valid_lo_ids or [])
    for new_id, item in enumerate(items, start=1):
        if item.get("lo_id") not in valid_lo_ids:
            continue
        cloned = dict(item)
        cloned["id"] = new_id
        normalized.append(cloned)
    return normalized


def get_item_score(item):
    hodnotenie = item.get("hodnotenie", {})
    if isinstance(hodnotenie, dict):
        score = hodnotenie.get("skore")
    else:
        score = item.get("hodnotenie_skore")
    try:
        return int(score)
    except (TypeError, ValueError):
        return None


def extract_prefixed_ids(errors, prefix):
    ids = set()
    pattern = re.compile(rf"{re.escape(prefix)}\s+(\d+)")
    for error in errors:
        match = pattern.search(str(error))
        if match:
            ids.add(int(match.group(1)))
    return ids
