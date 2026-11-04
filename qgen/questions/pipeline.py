import time
import re

from context_builder import (
    build_allowed_source_refs,
    build_page_map,
    build_context_for_lo,
    build_source_name_map,
    format_segment_label,
    parse_source_refs,
    resolve_source_names,
)
from document_language import detect_document_language
from item_answerability import analyze_item_answerability
from item_faithfulness import analyze_item_faithfulness
from item_relevance_to_lo import analyze_item_relevance_to_lo
from item_validation import validate_items
from json_load import safe_load_json
from llm_client import generate_with_retry
from outputs import (
    save_item_faithfulness_report,
    save_item_answerability_report,
    save_item_relevance_to_lo_report,
    save_processing_time_report,
    save_item_validation_report,
    save_python_code_correctness_report,
    save_python_code_runtime_report,
    save_python_code_syntax_report,
)
from python_code_eval import evaluate_python_code_items

ITEM_MIN_SCORE = 3
ITEM_MIN_ANSWERABILITY_SCORE = 3
ITEM_MIN_FAITHFULNESS_SCORE = 3
ITEM_DOCUMENT_CLASSIFICATION_MAX_CHARS = 12000


def generate_items_for_batch(
    los_batch,
    page_map,
    document_type_info=None,
    document_language="sk",
    model="gemini-2.5-flash-lite",
    client=None,
    verbose=True
):
    lo_blocks = []
    for lo in los_batch:
        lo_id = lo.get("id")
        context = build_context_for_lo(lo, page_map)
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
    document_type_info = document_type_info or {"is_python_document": False, "reason": ""}
    python_document = bool(document_type_info.get("is_python_document", False))
    document_type_reason = str(document_type_info.get("reason", "")).strip()

    prompt = build_item_generation_prompt(
        los_text,
        python_document=python_document,
        document_type_reason=document_type_reason,
        document_language=document_language,
    )

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        ids = [b["id"] for b in lo_blocks]
        if verbose:
            print(f"Generovanie položiek pre batch LO {ids} zlyhalo: {e}")
        return []

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


def evaluate_items_batch(items_batch, model="gemini-2.5-flash-lite", client=None, verbose=True, document_language="sk"):
    if not items_batch:
        return {}

    parts = []
    for item in items_batch:
        parts.append(
            f"Item id: {item.get('id')}\n"
            f"- lo_id: {item.get('lo_id')}\n"
            f"- typ: {item.get('typ', '')}\n"
            f"- otazka: {item.get('otazka', '')}\n"
            f"- odpoved: {item.get('odpoved', '')}\n"
            f"- jazyk: {item.get('jazyk', '')}\n"
            f"- execution_mode: {item.get('execution_mode', '')}\n"
            f"- function_name: {item.get('function_name', '')}\n"
            f"- kod_riesenia: {item.get('kod_riesenia', '')}\n"
            f"- test_cases: {item.get('test_cases', [])}\n"
            f"- napoveda: {item.get('napoveda', '')}\n"
            f"- citovane_zdroje: {item.get('citovane_zdroje', [])}"
        )
    items_text = "\n\n".join(parts)

    prompt = build_item_evaluation_prompt(items_text, document_language)

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Hodnotenie batchu položiek zlyhalo: {e}")
        return {}

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
    batch_size=10,
    evaluation_batch_size=20,
    model="gemini-2.5-flash-lite",
    generation_model=None,
    evaluation_model=None,
    output_dir=None,
    client=None,
    verbose=True,
    max_batch_attempts=3,
    max_eval_attempts=2,
    return_metrics=False,
):
    generation_model = generation_model or model
    evaluation_model = evaluation_model or model

    page_map = build_page_map(segmenty)
    source_name_map = build_source_name_map(segmenty)
    if verbose:
        print("Item pipeline: detekujem jazyk dokumentu.")
    language_info = detect_document_language(
        segmenty,
        client=client,
        model=generation_model,
        verbose=verbose,
    )
    document_language = language_info.get("language", "sk")
    if verbose:
        print("Item pipeline: klasifikujem dokument pre Python polozky.")
    document_type_info = classify_document_for_python_items(
        segmenty,
        client=client,
        model=evaluation_model,
        verbose=verbose,
    )
    lo_order = {lo.get("id"): idx for idx, lo in enumerate(los)}
    all_items = []
    next_item_id = 1
    total_los = len(los)

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
                document_type_info=document_type_info,
                document_language=document_language,
                model=generation_model,
                client=client,
                verbose=verbose
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
                    "jazyk": raw.get("jazyk", ""),
                    "kod_riesenia": raw.get("kod_riesenia", ""),
                    "execution_mode": raw.get("execution_mode", ""),
                    "function_name": raw.get("function_name", ""),
                    "automaticky_testovatelna": raw.get("automaticky_testovatelna", False),
                    "test_cases": raw.get("test_cases", []),
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
                    client=client,
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
        batch_num += 1
    all_items.sort(key=lambda item: item_sort_key(item, lo_order))
    for i, item in enumerate(all_items, start=1):
        item["id"] = i
    attach_source_names(all_items, source_name_map)

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
        client=client,
        verbose=verbose,
    )
    item_faithfulness_report = analyze_item_faithfulness(
        segmenty,
        all_items,
        client=client,
        verbose=verbose,
        batch_size=evaluation_batch_size,
        document_language=document_language,
    )
    item_answerability_report = analyze_item_answerability(
        segmenty,
        all_items,
        client=client,
        verbose=verbose,
        batch_size=evaluation_batch_size,
        document_language=document_language,
    )
    syntax_report, runtime_report, correctness_report = evaluate_python_code_items(all_items)
    accepted_items = filter_items_variant_b(
        all_items,
        item_validation_report,
        item_faithfulness_report,
        item_answerability_report,
        syntax_report,
        runtime_report,
        correctness_report,
    )
    normalized_items = normalize_accepted_items(accepted_items, valid_lo_ids)
    attach_source_names(normalized_items, source_name_map)

    if output_dir:
        save_item_validation_report(item_validation_report, output_dir)
        save_item_relevance_to_lo_report(item_relevance_report, output_dir)
        save_item_faithfulness_report(item_faithfulness_report, output_dir)
        save_item_answerability_report(item_answerability_report, output_dir)
        save_python_code_syntax_report(syntax_report, output_dir)
        save_python_code_runtime_report(runtime_report, output_dir)
        save_python_code_correctness_report(correctness_report, output_dir)
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
            "is_python_document": document_type_info.get("is_python_document", False),
            "document_type_reason": document_type_info.get("reason", ""),
            "document_language": document_language,
            "document_language_reason": language_info.get("reason", ""),
        },
    }
    if output_dir:
        save_processing_time_report(timing_report, output_dir, "item_processing_time_report.txt")

    if verbose:
        print(f"\nGenerovanie položiek pre všetky LO dokončené. Celkový počet položiek: {len(all_items)}")
        print(f"Cas generovania položiek: {generation_seconds:.2f} s")
        print(f"Cas evaluacie položiek: {evaluation_seconds:.2f} s")
    if return_metrics:
        timing_report["document_type_info"] = document_type_info
        timing_report["language_info"] = language_info
        timing_report["all_items"] = all_items
        return normalized_items, timing_report
    return normalized_items


def build_item_generation_prompt(los_text, python_document, document_type_reason, document_language):
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
- is_python_document: {"true" if python_document else "false"}
- reason: {document_type_reason or "no justification"}

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
- is_python_document: {"true" if python_document else "false"}
- reason: {document_type_reason or "bez zdovodnenia"}

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
        "jazyk": str(item.get("jazyk", "")).strip().lower(),
        "kod_riesenia": str(item.get("kod_riesenia", "")).strip(),
        "execution_mode": str(item.get("execution_mode", "")).strip(),
        "function_name": str(item.get("function_name", "")).strip(),
        "automaticky_testovatelna": bool(item.get("automaticky_testovatelna", False)),
        "test_cases": normalize_test_cases(item.get("test_cases", [])),
    }

    if normalized["typ"] != "prakticka_uloha":
        normalized["jazyk"] = ""
        normalized["kod_riesenia"] = ""
        normalized["execution_mode"] = ""
        normalized["function_name"] = ""
        normalized["automaticky_testovatelna"] = False
        normalized["test_cases"] = []
        return normalized

    if normalized["jazyk"] != "python":
        normalized["kod_riesenia"] = normalized["kod_riesenia"] or ""
        normalized["execution_mode"] = ""
        normalized["function_name"] = ""
        normalized["automaticky_testovatelna"] = False
        normalized["test_cases"] = []
        return normalized

    if normalized["execution_mode"] not in {"stdin_stdout", "function"}:
        normalized["execution_mode"] = ""
    if normalized["execution_mode"] != "function":
        normalized["function_name"] = ""
    if (
        not normalized["automaticky_testovatelna"]
        or not normalized["execution_mode"]
        or not normalized["test_cases"]
        or looks_non_testable_python_task(normalized)
        or has_invalid_function_mode_code(normalized)
    ):
        normalized["automaticky_testovatelna"] = False
        normalized["execution_mode"] = ""
        normalized["function_name"] = ""
        normalized["test_cases"] = []
    return normalized


def normalize_sources(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_test_cases(value):
    if not isinstance(value, list):
        return []

    normalized = []
    for row in value:
        if not isinstance(row, dict):
            continue
        normalized.append({
            "input": row.get("input", ""),
            "expected_output": row.get("expected_output", ""),
        })
    return normalized


def looks_non_testable_python_task(item):
    text = " ".join([
        item.get("otazka", ""),
        item.get("odpoved", "") if isinstance(item.get("odpoved", ""), str) else "",
        item.get("kod_riesenia", ""),
    ]).lower()
    blocked_tokens = [
        "import tkinter",
        "from tkinter",
        "customtkinter",
        "import turtle",
        "from turtle",
        "import pygame",
        "from pygame",
        "import kivy",
        "from kivy",
        "pyqt",
        "streamlit",
        "flask",
        "fastapi",
        "web app",
        "gui",
        "graficke rozhranie",
    ]
    return any(token in text for token in blocked_tokens)


def has_invalid_function_mode_code(item):
    if item.get("execution_mode") != "function":
        return False

    code = str(item.get("kod_riesenia", "")).strip()
    function_name = str(item.get("function_name", "")).strip()
    if not code or not function_name:
        return True

    try:
        import ast
        tree = ast.parse(code)
    except SyntaxError:
        return False

    found_target_function = False
    allowed_constant_values = (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                found_target_function = True
            continue
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.Assign):
            if isinstance(node.value, allowed_constant_values):
                continue
            return True
        if isinstance(node, ast.AnnAssign):
            if node.value is None or isinstance(node.value, allowed_constant_values):
                continue
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        return True

    return not found_target_function


def is_python_practical_item(item):
    return (
        item.get("typ") == "prakticka_uloha"
        and str(item.get("jazyk", "")).strip().lower() == "python"
        and str(item.get("kod_riesenia", "")).strip()
    )


def filter_items_variant_b(
    items,
    validation_report,
    faithfulness_report,
    answerability_report,
    syntax_report,
    runtime_report,
    correctness_report,
):
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
    syntax_by_id = {
        row.get("item_id"): row.get("syntax_valid")
        for row in syntax_report.get("items", [])
        if row.get("item_id") is not None
    }
    runtime_by_id = {
        row.get("item_id"): row.get("runtime_valid")
        for row in runtime_report.get("items", [])
        if row.get("item_id") is not None
    }
    correctness_by_id = {
        row.get("item_id"): row
        for row in correctness_report.get("items", [])
        if row.get("item_id") is not None
    }

    accepted = []
    for item in items:
        item_id = item.get("id")
        if item_id in invalid_item_ids:
            continue

        faithfulness_score = faithfulness_by_id.get(item_id)
        if faithfulness_score is None or faithfulness_score < ITEM_MIN_FAITHFULNESS_SCORE:
            continue

        score = get_item_score(item)
        answerability_score = answerability_by_id.get(item_id)
        score_ok = score is not None and score >= ITEM_MIN_SCORE
        answerability_ok = answerability_score is not None and answerability_score >= ITEM_MIN_ANSWERABILITY_SCORE
        if not (score_ok or answerability_ok):
            continue

        if is_python_practical_item(item):
            if syntax_by_id.get(item_id) is not True:
                continue
            if runtime_by_id.get(item_id) is not True:
                continue
            correctness_row = correctness_by_id.get(item_id, {})
            total = correctness_row.get("test_cases_total", 0)
            passed = correctness_row.get("test_cases_passed", 0)
            if total <= 0:
                continue
            if passed <= 0:
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


def classify_document_for_python_items(segmenty, client=None, model="gemini-2.5-flash-lite", verbose=True):
    source_text = build_full_document_text(segmenty, max_chars=ITEM_DOCUMENT_CLASSIFICATION_MAX_CHARS)
    if not source_text.strip():
        return {"is_python_document": False, "reason": "Dokument nema text pre klasifikaciu."}

    prompt = f"""
Posud nasledujuci dokument a rozhodni, ci je to dokument o programovacom jazyku Python.

Vrat LEN validny JSON v tvare:
{{
  "is_python_document": true,
  "reason": "stručné zdôvodnenie"
}}

Pravidla:
- true vrat len vtedy, ak je hlavnou temou dokumentu Python alebo vyucba programovania v jazyku Python
- false vrat pri ekonomike, matematike, fyzike, vseobecnej informatike alebo inom ne-Python dokumente
- false vrat aj vtedy, ak sa slovo Python spomenie len okrajovo

Dokument:
\"\"\"{source_text}\"\"\"
"""

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Klasifikacia dokumentu pre Python polozky zlyhala: {e}")
        return {"is_python_document": False, "reason": "Klasifikacia zlyhala, pouzity bezpecny fallback false."}

    if not isinstance(parsed, dict):
        return {"is_python_document": False, "reason": "Neplatna odpoved klasifikacie, pouzity bezpecny fallback false."}

    return {
        "is_python_document": bool(parsed.get("is_python_document", False)),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def build_full_document_text(segmenty, max_chars=None):
    parts = []
    total = 0
    for seg in segmenty:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if max_chars is not None:
            remaining = max_chars - total
            if remaining <= 0:
                break
            text = text[:remaining]
        page = seg.get("page")
        block = f"[{format_segment_label(seg)}]\n{text}" if page is not None else text
        parts.append(block)
        total += len(text)
    return "\n\n".join(parts)
