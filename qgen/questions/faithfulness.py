from qgen.context_builder import build_context_for_sources, build_page_map
from qgen.config import Settings, setting_or_default
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_Q_EVALUATION, batch_count, report_progress
from qgen.questions.prompts import question_record_text


def analyze_item_faithfulness(
    segmenty,
    items,
    llm=None,
    model=None,
    verbose: bool = True,
    batch_size=None,
    max_batch_attempts=None,
    document_language: str = "sk",
    context_max_chars=None,
    faithful_score=None,
    progress_callback=None,
    *,
    settings: Settings,
):
    model = setting_or_default(model, "llm_model", settings)
    batch_size = setting_or_default(batch_size, "question_eval_batch_size", settings)
    max_batch_attempts = setting_or_default(max_batch_attempts, "check_batch_attempts", settings)
    faithful_score = setting_or_default(faithful_score, "question_faithful_score", settings)
    report = {
        "stats": {
            "items_total": len(items),
            "items_evaluated": 0,
            "average_faithfulness_score": 0.0,
            "faithful_items": 0,
            "faithful_items_percent": 0.0,
        },
        "items": [],
    }

    if not items:
        return report

    page_map = build_page_map(segmenty)
    scores = []
    faithful_items = 0
    comparable_items = []

    for item in items:
        row = {
            "item_id": item.get("id"),
            "lo_id": item.get("lo_id"),
            "source_pages": item.get("source", []),
            "faithfulness_score": None,
            "faithful": False,
            "reason": "",
            "choices_ok": None,
            "wrong_choices": [],
        }
        report["items"].append(row)

        source_text = build_context_for_item(item, page_map, max_chars=context_max_chars)
        if not source_text.strip():
            continue

        comparable_items.append({
            "row": row,
            "item_id": item.get("id"),
            "item_text": item_to_text(item, document_language),
            "source_text": source_text,
        })

    total_batches = batch_count(len(comparable_items), batch_size)
    report_progress(progress_callback, STAGE_Q_EVALUATION, 0, total_batches)
    for start in range(0, len(comparable_items), batch_size):
        batch = comparable_items[start:start + batch_size]
        evaluations = {}
        for attempt in range(1, max_batch_attempts + 1):
            evaluations = evaluate_item_faithfulness_batch(
                batch,
                llm=llm,
                model=model,
                verbose=verbose,
                document_language=document_language,
                settings=settings,
            )
            if evaluations:
                break
            if verbose and max_batch_attempts > 1:
                print(f"Item faithfulness batch prazdny, opakujem ({attempt}/{max_batch_attempts})")

        for item in batch:
            evaluation = evaluations.get(item["item_id"])
            if not evaluation:
                continue
            row = item["row"]
            row["faithfulness_score"] = evaluation.get("skore")
            row["reason"] = evaluation.get("zdovodnenie", "")
            row["choices_ok"] = evaluation.get("choices_ok")
            row["wrong_choices"] = evaluation.get("wrong_choices", [])
            if isinstance(row["faithfulness_score"], int):
                scores.append(row["faithfulness_score"])
                row["faithful"] = row["faithfulness_score"] >= faithful_score
                if row["faithful"]:
                    faithful_items += 1
        report_progress(progress_callback, STAGE_Q_EVALUATION, start // batch_size + 1, total_batches)

    report["stats"]["items_evaluated"] = len(scores)
    if scores:
        report["stats"]["average_faithfulness_score"] = round(sum(scores) / len(scores), 4)
        report["stats"]["faithful_items"] = faithful_items
        report["stats"]["faithful_items_percent"] = round((faithful_items / len(scores)) * 100, 2)
    return report


def evaluate_item_faithfulness_batch(
    batch, llm=None, model=None, verbose: bool = True, document_language: str = "sk", *, settings: Settings
):
    labels = ("Question and answer", "Source") if document_language == "en" else ("Otázka a odpoveď", "Zdroj")
    parts = []
    for item in batch:
        parts.append(
            f"ITEM ID: {item['item_id']}\n"
            f"{labels[0]}:\n\"\"\"{item['item_text']}\"\"\"\n\n"
            f"{labels[1]}:\n\"\"\"{item['source_text']}\"\"\""
        )
    joined = "\n\n-----\n\n".join(parts)
    template = FAITHFULNESS_EN if document_language == "en" else FAITHFULNESS_SK
    prompt = template.replace("<<records>>", joined)

    # Step 6: no `except Exception: return None` (the questions then lost their score and
    # were filtered out); LLMError propagates. Step 13: the answer is {"items": [...]}.
    parsed = generate_json(
        llm, prompt, role="check", model=model, verbose=verbose, settings=settings, items_key="items"
    )

    evaluations = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        try:
            score = int(row.get("skore"))
        except (TypeError, ValueError):
            continue
        reason = str(row.get("zdovodnenie", "")).strip()
        choices_ok = row.get("choices_ok")
        wrong = row.get("wrong_choices")
        evaluations[item_id] = {
            "skore": max(1, min(5, score)),
            "zdovodnenie": reason,
            # Missing or not a bool = not checked (counts as ok in the filter).
            "choices_ok": choices_ok if isinstance(choices_ok, bool) else None,
            "wrong_choices": [str(w).strip() for w in wrong if str(w).strip()] if isinstance(wrong, list) else [],
        }
    return evaluations


def item_to_text(item, document_language: str = "sk"):
    """Type, question and answer; choice options marked [správna] / [nesprávna]."""
    return question_record_text(item, document_language, marks=True)


def build_context_for_item(item, page_map, max_chars=None):
    return build_context_for_sources(item.get("source", []), page_map, max_chars=max_chars)


FAITHFULNESS_SK = """
Posúď vernosť (faithfulness) každej otázky voči jej zdroju.

Skóre 1–5:
1 = odpoveď alebo zadanie obsahuje tvrdenia, ktoré zdroj nepodporuje
5 = všetko je plne podložené zdrojom
Pri "choice" posudzuj správne možnosti. Pri "no_answer" (úloha bez odpovede) posúď, či zadanie vychádza z učiva zdroja.

Pri "choice" navyše skontroluj označenie každej možnosti: "choices_ok" je true, ak každá možnosť označená [správna] je podľa zdroja naozaj správna a každá [nesprávna] je naozaj nesprávna. Inak je "choices_ok" false a "wrong_choices" obsahuje písmená zle označených možností. Pri ostatných typoch je "choices_ok" true a "wrong_choices" [].

Záznamy:
<<records>>

Vráť LEN validný JSON objekt bez ďalšieho textu:
{"items": [{"item_id": 1, "skore": 1, "zdovodnenie": "krátke zdôvodnenie", "choices_ok": true, "wrong_choices": []}]}
"""

FAITHFULNESS_EN = """
Assess the faithfulness of each question against its source.

Score 1-5:
1 = the answer or the question contains claims the source does not support
5 = everything is fully supported by the source
For "choice" assess the correct options. For "no_answer" (a task without an answer) assess whether the task is based on the subject matter of the source.

For "choice" also check the marking of every option: "choices_ok" is true when every option marked [correct] really is correct according to the source and every option marked [wrong] really is wrong. Otherwise "choices_ok" is false and "wrong_choices" lists the letters of the wrongly marked options. For the other types "choices_ok" is true and "wrong_choices" is [].

Records:
<<records>>

Return ONLY a valid JSON object with no extra text:
{"items": [{"item_id": 1, "skore": 1, "zdovodnenie": "short justification in English", "choices_ok": true, "wrong_choices": []}]}
"""
