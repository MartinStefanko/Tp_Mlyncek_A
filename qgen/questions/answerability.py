from qgen.context_builder import build_context_for_sources, build_page_map
from qgen.config import Settings, setting_or_default
from qgen.llm.parallel import run_batches
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_Q_EVALUATION, batch_count, counting_progress, report_progress
from qgen.questions.prompts import question_record_text


def analyze_item_answerability(
    segmenty,
    items,
    llm=None,
    model=None,
    verbose: bool = True,
    batch_size=None,
    max_batch_attempts=None,
    document_language: str = "sk",
    context_max_chars=None,
    answerable_score=None,
    progress_callback=None,
    *,
    settings: Settings,
):
    model = setting_or_default(model, "llm_model", settings)
    batch_size = setting_or_default(batch_size, "question_eval_batch_size", settings)
    max_batch_attempts = setting_or_default(max_batch_attempts, "check_batch_attempts", settings)
    answerable_score = setting_or_default(answerable_score, "question_answerable_score", settings)
    report = {
        "stats": {
            "items_total": len(items),
            "items_evaluated": 0,
            "average_answerability_score": 0.0,
            "answerable_items": 0,
            "answerable_items_percent": 0.0,
        },
        "items": [],
    }

    if not items:
        return report

    page_map = build_page_map(segmenty)
    scores = []
    answerable_items = 0
    comparable_items = []

    for item in items:
        row = {
            "item_id": item.get("id"),
            "lo_id": item.get("lo_id"),
            "source_pages": item.get("source", []),
            "answerability_score": None,
            "answerable": False,
            "reason": "",
        }
        report["items"].append(row)

        source_text = build_context_for_item(item, page_map, max_chars=context_max_chars)
        if not source_text.strip():
            continue

        comparable_items.append({
            "row": row,
            "item_id": item.get("id"),
            "question_text": question_to_text(item, document_language),
            "source_text": source_text,
        })

    total_batches = batch_count(len(comparable_items), batch_size)
    report_progress(progress_callback, STAGE_Q_EVALUATION, 0, total_batches)

    def evaluate_batch(batch):
        evaluations = {}
        for attempt in range(1, max_batch_attempts + 1):
            evaluations = evaluate_item_answerability_batch(
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
                print(f"Item answerability batch prazdny, opakujem ({attempt}/{max_batch_attempts})")
        if not isinstance(evaluations, dict):
            evaluations = {}
        return evaluations

    # Step 13b: the batches run at the same time; the results are merged in their order.
    batches = [comparable_items[start:start + batch_size] for start in range(0, len(comparable_items), batch_size)]
    all_evaluations = run_batches(
        [lambda batch=batch: evaluate_batch(batch) for batch in batches],
        settings=settings,
        on_done=counting_progress(progress_callback, STAGE_Q_EVALUATION, total_batches),
    )
    for batch, evaluations in zip(batches, all_evaluations):
        for item in batch:
            evaluation = evaluations.get(item["item_id"])
            if not evaluation:
                continue
            row = item["row"]
            row["answerability_score"] = evaluation.get("skore")
            row["reason"] = evaluation.get("zdovodnenie", "")
            if isinstance(row["answerability_score"], int):
                scores.append(row["answerability_score"])
                row["answerable"] = row["answerability_score"] >= answerable_score
                if row["answerable"]:
                    answerable_items += 1

    report["stats"]["items_evaluated"] = len(scores)
    if scores:
        report["stats"]["average_answerability_score"] = round(sum(scores) / len(scores), 4)
        report["stats"]["answerable_items"] = answerable_items
        report["stats"]["answerable_items_percent"] = round((answerable_items / len(scores)) * 100, 2)
    return report


def evaluate_item_answerability_batch(
    batch, llm=None, model=None, verbose: bool = True, document_language: str = "sk", *, settings: Settings
):
    labels = ("Question", "Source") if document_language == "en" else ("Otázka", "Zdroj")
    parts = []
    for item in batch:
        parts.append(
            f"ITEM ID: {item['item_id']}\n"
            f"{labels[0]}:\n\"\"\"{item['question_text']}\"\"\"\n\n"
            f"{labels[1]}:\n\"\"\"{item['source_text']}\"\"\""
        )
    joined = "\n\n-----\n\n".join(parts)
    template = ANSWERABILITY_EN if document_language == "en" else ANSWERABILITY_SK
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
        evaluations[item_id] = {"skore": max(1, min(5, score)), "zdovodnenie": reason}
    return evaluations


def question_to_text(item, document_language: str = "sk"):
    """Type, question and (for choice) the options without marks; no answer."""
    return question_record_text(item, document_language, marks=False)


def build_context_for_item(item, page_map, max_chars=None):
    return build_context_for_sources(item.get("source", []), page_map, max_chars=max_chars)


ANSWERABILITY_SK = """
Posúď zodpovedateľnosť (answerability) každej otázky voči jej zdroju.

Skóre 1–5:
1 = na otázku sa zo zdroja nedá spoľahlivo odpovedať
5 = na otázku sa dá zo zdroja odpovedať jasne a jednoznačne
Pri "choice" posúď, či sa zo zdroja dá jednoznačne určiť, ktoré možnosti sú správne.
Pri "no_answer" (úloha bez vyhodnotenia) posúď, či je zadanie jasné, splniteľné a súvisí s učivom zdroja.

Záznamy:
<<records>>

Vráť LEN validný JSON objekt bez ďalšieho textu:
{"items": [{"item_id": 1, "skore": 1, "zdovodnenie": "krátke zdôvodnenie"}]}
"""

ANSWERABILITY_EN = """
Assess the answerability of each question against its source.

Score 1-5:
1 = the question cannot be reliably answered from the source
5 = the question can be answered clearly and unambiguously from the source
For "choice" assess whether the source makes it unambiguous which options are correct.
For "no_answer" (a task that is not graded) assess whether the task is clear, doable and related to the subject matter of the source.

Records:
<<records>>

Return ONLY a valid JSON object with no extra text:
{"items": [{"item_id": 1, "skore": 1, "zdovodnenie": "short justification in English"}]}
"""
