"""Question pipeline (krok 13): questions of type choice / open / no_answer, as many as the
content of each section (tematický celok) allows.

    1. language of the document (as before)
    2. per batch of QGEN_QUESTION_GENERATION_BATCH_SIZE sections:
         first call (repeated up to question_generation_attempts times when empty),
         then QGEN_QUESTION_EXTRA_ROUNDS extra rounds with the questions created so far;
         a section stops when a round brought nothing new
       every answer: normalization (qgen.questions.types), validation, type rules,
       duplicates within the section (qgen.questions.dedup)
       an answer cut off at the output token limit splits the batch in half
    3. evaluation of the kept questions: quality score (chunks of question_eval_batch_size),
       faithfulness with the check of the choice marking, answerability, filter
    4. coverage: sections with fewer accepted questions than per_section get one more
       call; the new questions go through 2-3 as well
    5. relevance to the section (all questions), ordering, ids 1..n

Step 13b: the calls of one step run at the same time (QGEN_LLM_MAX_CONCURRENCY,
qgen.llm.parallel): the first calls of all batches, then each extra round for all batches
(a round starts when every batch of the previous one has finished), the quality score
chunks, the faithfulness / answerability batches and the coverage calls. The answers are
processed in the main thread in the order of the batches; after the generation step the
questions get the ids of the former run (batch by batch, round by round), so the result
does not depend on the order in which the calls finish.

per_section (options.questions.per_section) is a minimum, never a target or a limit.
Nothing is written to disk. LLMError propagates (also when a single section does not fit
into the output token limit).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from qgen.config import Settings, setting_or_default
from qgen.contract.options import OptionDefaults, QuestionOptions
from qgen.context_builder import (
    build_allowed_source_refs,
    build_context_for_lo,
    build_page_map,
    build_source_name_map,
    parse_source_refs,
    resolve_source_names,
)
from qgen.extraction.document_language import resolve_document_language
from qgen.llm.base import EmbeddingClient, LLMClient
from qgen.llm.errors import LLMError
from qgen.llm.parallel import run_batches
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_Q_EVALUATION, STAGE_QUESTIONS, ProgressCallback, batch_count, report_progress
from qgen.questions.answerability import analyze_item_answerability
from qgen.questions.dedup import SectionDeduplicator
from qgen.questions.faithfulness import analyze_item_faithfulness
from qgen.questions.prompts import (
    build_coverage_section,
    build_extra_round_section,
    build_item_evaluation_prompt,
    build_item_generation_prompt,
    build_lo_blocks,
    evaluation_items_text,
)
from qgen.questions.relevance_to_lo import analyze_item_relevance_to_lo
from qgen.questions.types import (
    ORIGIN_COVERAGE,
    ORIGIN_INITIAL,
    QUESTION_TYPES,
    QuestionSettings,
    extra_origin,
    normalize_generated_item,
    normalize_sources,
    type_permission_problem,
)
from qgen.questions.validation import WARNING_PAGE_REFERENCE, validate_items

__all__ = [
    "QuestionPipelineResult",
    "generate_all_items",
    "generate_items_for_batch",
    "evaluate_items_batch",
    "build_item_generation_prompt",
    "build_item_evaluation_prompt",
    "filter_items_variant_b",
    "normalize_sources",
]


@dataclass
class QuestionPipelineResult:
    """Result of generate_all_items. Nothing is written to disk.

    questions      accepted questions with ids 1..n (internal format, qgen.questions.types)
    all_questions  all generated questions (ids 1..n of this list) with `drop_reason`
                   (None = accepted)
    language_info  {"language": "sk"|"en", "reason": str}
    reports        report dicts, keys: item_validation, item_relevance_to_lo,
                   item_faithfulness, item_answerability, item_processing_time,
                   question_diagnostics
    """

    questions: list[dict]
    all_questions: list[dict]
    language_info: dict[str, str]
    reports: dict[str, Any] = field(default_factory=dict)


DEFAULT_PER_SECTION = OptionDefaults.per_section  # options.questions.per_section (Príloha A)

# Reasons of dropped questions in the question_diagnostics report.
DROP_MISSING_LO_ID = "missing_lo_id"              # generated item without "lo_id"
DROP_INVALID_ITEM_SHAPE = "invalid_item_shape"    # array element that is not a JSON object
DROP_VALIDATION = "validation"                    # failed validate_items
DROP_TYPE_NOT_ALLOWED = "type_not_allowed"        # type not in options.questions.types
DROP_TYPE_RULE = "type_rule"                      # no_answer at a Bloom level that does not allow it
DROP_DUPLICATE = "duplicate"                      # too similar to a question of the same section
DROP_LOW_SCORE = "low_score"                      # quality / faithfulness / answerability filter
DROP_WRONG_CHOICE_MARKING = "wrong_choice_marking"  # faithfulness check: is_correct does not match
DROP_UNKNOWN_LO_ID = "unknown_lo_id"              # lo_id is not one of the sections
DROP_REASONS = (
    DROP_MISSING_LO_ID, DROP_INVALID_ITEM_SHAPE, DROP_VALIDATION, DROP_TYPE_NOT_ALLOWED, DROP_TYPE_RULE,
    DROP_DUPLICATE, DROP_LOW_SCORE, DROP_WRONG_CHOICE_MARKING, DROP_UNKNOWN_LO_ID,
)

STATUS_PENDING = "pending"    # kept after generation, waiting for the evaluation
STATUS_ACCEPTED = "accepted"

WARNING_SECTION_WITHOUT_QUESTIONS = "section_without_questions"
WARNING_CHOICE_MARKING = "choice_marking_wrong"
WARNING_CHOICE_SHARE = "choice_share_low"


@dataclass
class BatchItems:
    """Answer of one question generation call (or of the halves of a split call).

    items       raw items with lo_id
    dropped     dropped items by reason (DROP_MISSING_LO_ID, DROP_INVALID_ITEM_SHAPE)
    no_content  lo_id -> reason, when the model said the section has no subject matter
                (object {"lo_id": .., "reason": ..} without a question)
    splits      how many times the batch was split because the answer was cut off
    calls       generation calls made
    """

    items: list[dict] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    no_content: dict = field(default_factory=dict)
    splits: int = 0
    calls: int = 0

    def extend(self, other: "BatchItems") -> None:
        self.items.extend(other.items)
        self.dropped.update(other.dropped)
        self.no_content.update(other.no_content)
        self.splits += other.splits
        self.calls += other.calls


def _is_no_content_marker(item: dict) -> bool:
    return "reason" in item and not str(item.get("description", "") or "").strip()


def _section_blocks(los_batch, page_map, context_max_chars, verbose):
    blocks = []
    for lo in los_batch:
        lo_id = lo.get("id")
        context = build_context_for_lo(lo, page_map, max_chars=context_max_chars)
        if not context.strip():
            if verbose:
                print(f"  LO {lo_id}: bez kontextu, preskakujem v tomto batche.")
            continue
        blocks.append({
            "id": lo_id,
            "name": lo.get("vzdelávací_objekt", ""),
            "bloom": lo.get("bloom_level", ""),
            "context": context,
        })
    return blocks


def generate_items_for_batch(
    los_batch,
    page_map,
    document_language="sk",
    model=None,
    llm=None,
    verbose=True,
    context_max_chars=None,
    *,
    settings: Settings,
    question_settings: Optional[QuestionSettings] = None,
    per_section: Optional[int] = None,
    existing: Optional[dict] = None,
    missing: Optional[dict] = None,
) -> BatchItems:
    """One generation call for the sections of `los_batch`.

    existing=None: first call. existing={lo_id: [questions]}: extra round (missing=None) or
    coverage call (missing={lo_id: count}); only the sections of this batch are listed.
    An answer cut off at the token limit raises LLMError with `truncated` = True.
    """
    model = setting_or_default(model, "llm_model_questions", settings)
    qs = question_settings or QuestionSettings.from_options(None, per_section)
    blocks = _section_blocks(los_batch, page_map, context_max_chars, verbose)
    if not blocks:
        return BatchItems()

    extra_section = ""
    if existing is not None:
        ids = [block["id"] for block in blocks]
        listed = {lo_id: existing.get(lo_id, []) for lo_id in ids}
        if missing is None:
            extra_section = build_extra_round_section(listed, document_language)
        else:
            extra_section = build_coverage_section(listed, {i: missing.get(i, 1) for i in ids}, document_language)
    prompt = build_item_generation_prompt(
        build_lo_blocks(blocks, document_language), document_language, qs, settings, extra_section
    )

    # Step 6: no `except Exception: return []`; LLMError propagates. items_key: an answer
    # that is not {"items": [...]} is invalid JSON (repeated call, then LLMError).
    parsed = generate_json(
        llm, prompt, role="questions", model=model, verbose=verbose, settings=settings,
        items_key="items", max_tokens=settings.llm_max_output_tokens,
    )

    result = BatchItems(calls=1)
    for item in parsed:
        if not isinstance(item, dict):
            result.dropped[DROP_INVALID_ITEM_SHAPE] += 1
            continue
        if "lo_id" not in item:
            result.dropped[DROP_MISSING_LO_ID] += 1
            continue
        if _is_no_content_marker(item):
            if existing is None:
                result.no_content[item.get("lo_id")] = str(item.get("reason", "")).strip()
            continue
        result.items.append(item)
    return result


def generate_with_split(los_batch, page_map, *, verbose=False, **kwargs) -> BatchItems:
    """generate_items_for_batch; an answer cut off at the token limit splits the batch in
    half (recursively). A single section that does not fit raises the LLMError."""
    try:
        return generate_items_for_batch(los_batch, page_map, verbose=verbose, **kwargs)
    except LLMError as error:
        if not error.truncated or len(los_batch) <= 1:
            raise
    if verbose:
        print(f"Odpoved odrezana pri {len(los_batch)} celkoch, delim batch na polovicu.")
    half = (len(los_batch) + 1) // 2
    result = BatchItems(splits=1)
    result.extend(generate_with_split(los_batch[:half], page_map, verbose=verbose, **kwargs))
    result.extend(generate_with_split(los_batch[half:], page_map, verbose=verbose, **kwargs))
    return result


def evaluate_items_batch(items_batch, model=None, llm=None, verbose=True, document_language="sk", *, settings: Settings):
    """Quality score 1-5 of the questions (role eval): {id: {"skore", "zdovodnenie"}}."""
    if not items_batch:
        return {}
    model = setting_or_default(model, "llm_model_eval", settings)
    prompt = build_item_evaluation_prompt(evaluation_items_text(items_batch, document_language), document_language)

    # Step 6: no `except Exception: return {}`; LLMError propagates.
    parsed = generate_json(llm, prompt, role="eval", model=model, verbose=verbose, settings=settings, items_key="items")

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
    refs = parse_source_refs(item.get("source", []))
    source_id, first_page = refs[0] if refs else ("", float("inf"))
    return (lo_pos, source_id or "", first_page, item.get("id", float("inf")))


def attach_source_names(items, source_name_map):
    for item in items:
        item["source_files"] = resolve_source_names(item.get("source", []), source_name_map)
    return items


def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


class _Progress:
    """done / total of one step, monotonic; `estimate` units reserved for a sub-step."""

    def __init__(self, callback, stage, total):
        self.callback, self.stage, self.done, self.total = callback, stage, 0, total
        report_progress(callback, stage, 0, total)

    def grow(self, units):
        self.total += units

    def advance(self, units=1):
        self.done = min(self.total, self.done + units)
        report_progress(self.callback, self.stage, self.done, self.total)

    def jump(self, done):
        self.done = max(self.done, min(self.total, done))
        report_progress(self.callback, self.stage, self.done, self.total)

    def sub(self, estimate):
        start = self.done

        def callback(stage, done, _total):
            report_progress(self.callback, stage, start + min(done, estimate), self.total)

        return callback, start + estimate

    def finish(self):
        self.total = max(self.total, self.done)
        self.done = self.total
        report_progress(self.callback, self.stage, self.done, self.total)


class _QuestionRun:
    """State of one generate_all_items run."""

    def __init__(self, los, segmenty, *, llm, embedder, settings, qs, language, generation_model,
                 evaluation_model, batch_size, evaluation_batch_size, max_batch_attempts, max_eval_attempts,
                 extra_rounds, verbose, progress_callback):
        self.los = los
        self.segmenty = segmenty
        self.llm = llm
        self.embedder = embedder
        self.settings = settings
        self.qs = qs
        self.language = language
        self.generation_model = generation_model
        self.evaluation_model = evaluation_model
        self.batch_size = batch_size
        self.evaluation_batch_size = evaluation_batch_size
        self.max_batch_attempts = max_batch_attempts
        self.max_eval_attempts = max_eval_attempts
        self.extra_rounds = extra_rounds
        self.verbose = verbose
        self.progress_callback = progress_callback

        self.page_map = build_page_map(segmenty)
        self.source_name_map = build_source_name_map(segmenty)
        self.allowed_pages = build_allowed_source_refs(segmenty)
        self.sections = {lo.get("id"): lo for lo in los}
        self.valid_lo_ids = {lo.get("id") for lo in los if isinstance(lo.get("id"), int)}
        self.dedup = SectionDeduplicator(settings.question_dedup_threshold, embedder, settings=settings,
                                         verbose=verbose)

        self.items: list[dict] = []
        self.status: dict[int, str] = {}
        self.next_id = 1
        self.generation_dropped: Counter = Counter()
        self.no_content: dict = {}
        self.splits = 0
        self.calls: Counter = Counter()
        self.times_clamped = 0
        self.duplicates_by_origin: Counter = Counter()
        self.rounds: list[dict] = []
        self.wrong_marking: list[dict] = []
        self.generation_seconds = 0.0
        self.evaluation_seconds = 0.0
        # Generation step only: (batch, round) of every question and of every duplicate
        # count, to restore the order of the former batch-by-batch run (renumber_generation).
        self.position: dict[int, tuple[int, int]] = {}
        self.duplicate_events: list[tuple[tuple[int, int], str, int]] = []

    # --- generation ---------------------------------------------------------------------

    def call_generation(self, sections, existing=None, missing=None) -> BatchItems:
        """One generation call (split when cut off). Changes no state: runs in a worker thread."""
        return generate_with_split(
            sections,
            self.page_map,
            verbose=self.verbose,
            document_language=self.language,
            model=self.generation_model,
            llm=self.llm,
            context_max_chars=self.settings.context_max_chars,
            settings=self.settings,
            question_settings=self.qs,
            existing=existing,
            missing=missing,
        )

    def record_generation(self, result: BatchItems, origin: str) -> None:
        """Counters of one generation answer (main thread, in the order of the batches)."""
        self.generation_dropped.update(result.dropped)
        self.no_content.update(result.no_content)
        self.splits += result.splits
        self.calls[origin.split("_")[0]] += result.calls

    def accept_new(self, result: BatchItems, origin: str,
                   position: Optional[tuple[int, int]] = None) -> tuple[Counter, list[dict]]:
        """Normalize, validate and deduplicate the items of one answer (main thread).

        position: (batch, round) in the generation step (see renumber_generation).
        Returns (new questions per section, new questions); they wait for the evaluation."""
        records = []
        for raw in result.items:
            section = self.sections.get(raw.get("lo_id"))
            item, notes = normalize_generated_item(raw, section, self.qs, self.language)
            item = {"id": self.next_id, **item, "origin": origin, "evaluation": {"score": None, "reason": ""}}
            item["source_files"] = resolve_source_names(item["source"], self.source_name_map)
            if position is not None:
                self.position[item["id"]] = position
            self.next_id += 1
            self.times_clamped += int(notes["times_clamped"])
            records.append(item)
            self.items.append(item)

        report = validate_items(records, allowed_pages=self.allowed_pages, valid_lo_ids=self.valid_lo_ids,
                                question_settings=self.qs)
        invalid = extract_prefixed_ids(report.get("errors", []), "Polozka")
        candidates = []
        for item in records:
            if item["id"] in invalid:
                self.status[item["id"]] = DROP_VALIDATION
                continue
            problem = type_permission_problem(item, self.qs, self.settings)
            if problem:
                self.status[item["id"]] = problem
                continue
            candidates.append(item)

        new, duplicates = self.dedup.filter(candidates)
        for item in duplicates:
            self.status[item["id"]] = DROP_DUPLICATE
            self.duplicates_by_origin[origin] += 1
        if position is not None:
            self.duplicate_events.append((position, origin, len(duplicates)))
        for item in new:
            self.status[item["id"]] = STATUS_PENDING
        return Counter(item["lo_id"] for item in new), new

    def existing_for(self, sections) -> dict:
        ids = {lo.get("id") for lo in sections}
        existing: dict = {lo_id: [] for lo_id in ids}
        for item in self.items:
            if item.get("lo_id") in ids and item.get("description"):
                existing[item["lo_id"]].append(item)
        return existing

    def run_generation(self) -> list[dict]:
        """Step 2 for all batches; returns the questions waiting for the evaluation.

        Round 0 = the first calls of all batches, then the extra rounds; the calls of one
        round run at the same time, the next round starts when the round has finished."""
        started = time.perf_counter()
        batches = _chunks(self.los, self.batch_size)
        per_batch = 1 + self.extra_rounds
        progress = _Progress(self.progress_callback, STAGE_QUESTIONS, len(batches) * per_batch)
        if self.verbose:
            print(f"Začínam generovanie otázok pre {len(self.los)} celkov v batchoch po {self.batch_size}.")
        pending: list[dict] = []
        round_stats: dict[int, Counter] = {r: Counter() for r in range(1, self.extra_rounds + 1)}

        def first_call(number, batch):
            results = []
            for attempt in range(1, self.max_batch_attempts + 1):
                result = self.call_generation(batch)
                results.append(result)
                if result.items:
                    break
                if self.verbose and self.max_batch_attempts > 1:
                    print(f"Batch {number + 1}: prazdny vystup, opakujem ({attempt}/{self.max_batch_attempts})")
            return results

        answers = run_batches(
            [lambda number=number, batch=batch: first_call(number, batch) for number, batch in enumerate(batches)],
            settings=self.settings, on_done=lambda _index, _result: progress.advance(),
        )
        active_by_batch: dict[int, list] = {}
        for number, (batch, results) in enumerate(zip(batches, answers)):
            if self.verbose:
                print(f"\nBatch {number + 1}: LO id {[lo.get('id') for lo in batch]}")
            new_counts, new = Counter(), []
            for result in results:
                self.record_generation(result, ORIGIN_INITIAL)
                if result.items:
                    new_counts, new = self.accept_new(result, ORIGIN_INITIAL, position=(number, 0))
            pending.extend(new)
            active_by_batch[number] = [
                lo for lo in batch if new_counts[lo.get("id")] > 0 and lo.get("id") not in self.no_content
            ]

        for round_no in range(1, self.extra_rounds + 1):
            work = [(number, active) for number, active in active_by_batch.items() if active]
            if not work:
                break
            if len(work) < len(batches):
                progress.advance(len(batches) - len(work))  # batches that stopped: their unit is done
            origin = extra_origin(round_no)
            calls = [(active, self.existing_for(active)) for _, active in work]
            results = run_batches(
                [lambda active=active, existing=existing: self.call_generation(active, existing=existing)
                 for active, existing in calls],
                settings=self.settings, on_done=lambda _index, _result: progress.advance(),
            )
            for (number, active), result in zip(work, results):
                self.record_generation(result, origin)
                counts, new = self.accept_new(result, origin, position=(number, round_no))
                pending.extend(new)
                round_stats[round_no]["calls"] += result.calls
                round_stats[round_no]["sections"] += len(active)
                round_stats[round_no]["questions_added"] += len(new)
                active_by_batch[number] = [lo for lo in active if counts[lo.get("id")] > 0]
                round_stats[round_no]["sections_continuing"] += len(active_by_batch[number])

        progress.finish()
        self.rounds = [{"round": r, **{k: stats[k] for k in ("calls", "sections", "questions_added",
                                                              "sections_continuing")}}
                       for r, stats in round_stats.items()]
        pending = self.renumber_generation(pending)
        self.generation_seconds += time.perf_counter() - started
        return pending

    def renumber_generation(self, pending: list[dict]) -> list[dict]:
        """Ids, order and duplicate counters of the former batch-by-batch run.

        The rounds ran one after another over all batches (round-major); the former run
        went batch by batch (batch 1 with its extra rounds, then batch 2, ...). The ids
        go into the evaluation prompts, so they must be the same."""
        order = sorted(self.items, key=lambda item: (self.position[item["id"]], item["id"]))
        new_ids = {item["id"]: number for number, item in enumerate(order, start=1)}
        self.status = {new_ids[item["id"]]: self.status[item["id"]] for item in order if item["id"] in self.status}
        for item in order:
            item["id"] = new_ids[item["id"]]
        self.items = order
        self.position = {}
        duplicates: Counter = Counter()
        for _position, origin, count in sorted(self.duplicate_events, key=lambda event: event[0]):
            if count:
                duplicates[origin] += count
        self.duplicates_by_origin = duplicates
        self.duplicate_events = []
        return sorted(pending, key=lambda item: item["id"])

    # --- evaluation -----------------------------------------------------------------------

    def evaluate(self, pending: list[dict], progress: _Progress) -> tuple[dict, dict]:
        """Quality score, faithfulness (+ choice marking), answerability and the filter.
        Sets the status of every pending question to accepted or a drop reason."""
        started = time.perf_counter()

        def score_chunk(chunk):
            scores = {}
            for attempt in range(1, self.max_eval_attempts + 1):
                scores = evaluate_items_batch(chunk, model=self.evaluation_model, llm=self.llm, verbose=self.verbose,
                                              document_language=self.language, settings=self.settings)
                if scores:
                    break
                if self.verbose and self.max_eval_attempts > 1:
                    print(f"Hodnotenie prazdne, opakujem ({attempt}/{self.max_eval_attempts})")
            return scores

        chunks = _chunks(pending, self.evaluation_batch_size)
        all_scores = run_batches(
            [lambda chunk=chunk: score_chunk(chunk) for chunk in chunks],
            settings=self.settings, on_done=lambda _index, _result: progress.advance(),
        )
        for chunk, scores in zip(chunks, all_scores):
            for item in chunk:
                score = scores.get(item["id"], {})
                item["evaluation"] = {"score": score.get("skore"), "reason": score.get("zdovodnenie", "")}

        check_batches = batch_count(len(pending), self.evaluation_batch_size)
        callback, end = progress.sub(check_batches)
        faithfulness = analyze_item_faithfulness(
            self.segmenty, pending, llm=self.llm, model=self.settings.llm_model, verbose=self.verbose,
            batch_size=self.evaluation_batch_size, max_batch_attempts=self.settings.check_batch_attempts,
            document_language=self.language, context_max_chars=self.settings.context_max_chars,
            faithful_score=self.settings.question_faithful_score, progress_callback=callback, settings=self.settings,
        )
        progress.jump(end)
        callback, end = progress.sub(check_batches)
        answerability = analyze_item_answerability(
            self.segmenty, pending, llm=self.llm, model=self.settings.llm_model, verbose=self.verbose,
            batch_size=self.evaluation_batch_size, max_batch_attempts=self.settings.check_batch_attempts,
            document_language=self.language, context_max_chars=self.settings.context_max_chars,
            answerable_score=self.settings.question_answerable_score, progress_callback=callback,
            settings=self.settings,
        )
        progress.jump(end)

        marking = {row.get("item_id"): row for row in faithfulness.get("items", [])}
        to_filter = []
        for item in pending:
            row = marking.get(item["id"]) or {}
            if item.get("type") == "choice" and row.get("choices_ok") is False:
                self.status[item["id"]] = DROP_WRONG_CHOICE_MARKING
                self.wrong_marking.append({"item": item, "wrong_choices": row.get("wrong_choices", [])})
                continue
            to_filter.append(item)
        accepted = filter_items_variant_b(
            to_filter, {"errors": []}, faithfulness, answerability,
            min_score=self.settings.question_min_score,
            min_faithfulness_score=self.settings.question_min_faithfulness_score,
            min_answerability_score=self.settings.question_min_answerability_score,
        )
        accepted_ids = {item["id"] for item in accepted}
        for item in to_filter:
            self.status[item["id"]] = STATUS_ACCEPTED if item["id"] in accepted_ids else DROP_LOW_SCORE
        self.evaluation_seconds += time.perf_counter() - started
        return faithfulness, answerability

    def accepted_per_section(self) -> Counter:
        return Counter(item["lo_id"] for item in self.items if self.status.get(item["id"]) == STATUS_ACCEPTED)

    def evaluation_units(self, count: int) -> int:
        return batch_count(count, self.evaluation_batch_size) * 3


def merge_check_reports(reports: list[dict], kind: str) -> dict:
    """Join faithfulness or answerability reports of several phases; stats from the rows."""
    score_key, flag_key = f"{kind}_score", ("faithful" if kind == "faithfulness" else "answerable")
    rows = [row for report in reports for row in report.get("items", [])]
    scores = [row[score_key] for row in rows if isinstance(row.get(score_key), int)]
    flagged = sum(1 for row in rows if row.get(flag_key))
    return {
        "stats": {
            "items_total": sum(report.get("stats", {}).get("items_total", 0) for report in reports),
            "items_evaluated": len(scores),
            f"average_{kind}_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            f"{flag_key}_items": flagged if scores else 0,
            f"{flag_key}_items_percent": round(flagged / len(scores) * 100, 2) if scores else 0.0,
        },
        "items": rows,
    }


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
    *,
    settings: Settings,
    per_section: Optional[int] = None,
    question_options: Optional[QuestionOptions] = None,
    extra_rounds: Optional[int] = None,
) -> QuestionPipelineResult:
    """Question pipeline (see the module docstring).

    question_options: resolved options.questions (types, per_section, choice_options,
    hints); None = the defaults of Príloha A. per_section overrides its per_section.
    batch_size: sections per generation call (None = QGEN_QUESTION_GENERATION_BATCH_SIZE).
    extra_rounds: None = QGEN_QUESTION_EXTRA_ROUNDS.
    document_language: None or "auto" = detect with the LLM; "sk" / "en" = use it.
    llm / embedder = None: created by qgen.llm.factory per role.
    An LLM or embedding call that fails for good raises LLMError.
    """
    qs = QuestionSettings.from_options(question_options, per_section)
    batch_size = setting_or_default(batch_size, "question_generation_batch_size", settings)
    evaluation_batch_size = setting_or_default(evaluation_batch_size, "question_eval_batch_size", settings)
    max_batch_attempts = setting_or_default(max_batch_attempts, "question_generation_attempts", settings)
    max_eval_attempts = setting_or_default(max_eval_attempts, "question_eval_attempts", settings)
    extra_rounds = setting_or_default(extra_rounds, "question_extra_rounds", settings)
    generation_model = generation_model or model or settings.llm_model_questions
    evaluation_model = evaluation_model or model or settings.llm_model_eval

    if verbose:
        print("Item pipeline: detekujem jazyk dokumentu.")
    language_info = resolve_document_language(
        segmenty, document_language, llm=llm, model=generation_model, settings=settings, verbose=verbose,
        progress_callback=progress_callback, role="questions",
    )
    language = language_info.get("language", "sk")
    run = _QuestionRun(
        los, segmenty, llm=llm, embedder=embedder, settings=settings, qs=qs, language=language,
        generation_model=generation_model, evaluation_model=evaluation_model, batch_size=batch_size,
        evaluation_batch_size=evaluation_batch_size, max_batch_attempts=max_batch_attempts,
        max_eval_attempts=max_eval_attempts, extra_rounds=extra_rounds, verbose=verbose,
        progress_callback=progress_callback,
    )

    # 2. generation with extra rounds
    pending = run.run_generation()

    # 3. evaluation (+1 unit for the relevance at the end, +1 reserved for the coverage call)
    progress = _Progress(progress_callback, STAGE_Q_EVALUATION, run.evaluation_units(len(pending)) + 2)
    if verbose:
        print("Item pipeline: hodnotenie a kontroly otázok.")
    faithfulness_reports, answerability_reports = [], []
    faithfulness, answerability = run.evaluate(pending, progress)
    faithfulness_reports.append(faithfulness)
    answerability_reports.append(answerability)

    # 4. coverage: sections below per_section after the filter
    accepted_before = run.accepted_per_section()
    under_minimum = [lo for lo in los if accepted_before[lo.get("id")] < qs.per_section]
    coverage_calls = 0
    coverage_pending: list[dict] = []
    if under_minimum:
        if verbose:
            print(f"\nDoplnam otazky pre celky pod minimom: {[lo.get('id') for lo in under_minimum]}")
        chunks = _chunks(under_minimum, batch_size)
        progress.grow(len(chunks) - 1)
        started = time.perf_counter()
        calls = [
            (chunk, run.existing_for(chunk),
             {lo.get("id"): max(1, qs.per_section - accepted_before[lo.get("id")]) for lo in chunk})
            for chunk in chunks
        ]
        results = run_batches(
            [lambda chunk=chunk, existing=existing, missing=missing:
             run.call_generation(chunk, existing=existing, missing=missing)
             for chunk, existing, missing in calls],
            settings=settings, on_done=lambda _index, _result: progress.advance(),
        )
        for result in results:
            run.record_generation(result, ORIGIN_COVERAGE)
            coverage_calls += result.calls
            _, new = run.accept_new(result, ORIGIN_COVERAGE)
            coverage_pending.extend(new)
        run.generation_seconds += time.perf_counter() - started
        if coverage_pending:
            progress.grow(run.evaluation_units(len(coverage_pending)))
            faithfulness, answerability = run.evaluate(coverage_pending, progress)
            faithfulness_reports.append(faithfulness)
            answerability_reports.append(answerability)
    else:
        progress.advance()
    accepted_after = run.accepted_per_section()
    filled = [lo.get("id") for lo in under_minimum if accepted_after[lo.get("id")] > accepted_before[lo.get("id")]]

    # 5. relevance, order, ids
    started = time.perf_counter()
    lo_order = {lo.get("id"): idx for idx, lo in enumerate(los)}
    all_items = sorted(run.items, key=lambda item: item_sort_key(item, lo_order))
    new_ids = {item["id"]: position for position, item in enumerate(all_items, start=1)}
    status = {new_ids[old]: reason for old, reason in run.status.items()}
    for item in all_items:
        item["id"] = new_ids[item["id"]]
    attach_source_names(all_items, run.source_name_map)
    item_faithfulness_report = _renumber_rows(merge_check_reports(faithfulness_reports, "faithfulness"), new_ids)
    item_answerability_report = _renumber_rows(merge_check_reports(answerability_reports, "answerability"), new_ids)
    item_validation_report = validate_items(all_items, allowed_pages=run.allowed_pages,
                                            valid_lo_ids=run.valid_lo_ids, question_settings=qs)
    item_relevance_report = analyze_item_relevance_to_lo(
        all_items, los, embedder=embedder, embedding_model=settings.embed_model, verbose=verbose, settings=settings,
    )
    progress.finish()

    accepted_items = [item for item in all_items if status.get(item["id"]) == STATUS_ACCEPTED]
    normalized_items = normalize_accepted_items(accepted_items, run.valid_lo_ids)
    run.evaluation_seconds += time.perf_counter() - started
    all_questions = [{**item, "drop_reason": None if status.get(item["id"]) == STATUS_ACCEPTED
                      else status.get(item["id"])} for item in all_items]

    diagnostics_report = build_question_diagnostics(
        los, all_items, status, normalized_items, run, qs,
        under_minimum=[lo.get("id") for lo in under_minimum],
        coverage_calls=coverage_calls, sections_filled=filled,
        validation_report=item_validation_report,
    )
    timing_report = {
        "pipeline": "items",
        "generation_seconds": round(run.generation_seconds, 4),
        "evaluation_seconds": round(run.evaluation_seconds, 4),
        "total_seconds": round(run.generation_seconds + run.evaluation_seconds, 4),
        "details": {
            "items_count_all": len(all_items),
            "items_count_accepted": len(normalized_items),
            "los_count": len(los),
            "document_language": language,
            "document_language_reason": language_info.get("reason", ""),
        },
    }
    if verbose:
        print(f"\nGenerovanie otázok dokončené. Vygenerované: {len(all_items)}, prijaté: {len(normalized_items)}")
        print(f"Cas generovania: {run.generation_seconds:.2f} s, cas hodnotenia: {run.evaluation_seconds:.2f} s")
    return QuestionPipelineResult(
        questions=normalized_items,
        all_questions=all_questions,
        language_info=language_info,
        reports={
            "item_validation": item_validation_report,
            "item_relevance_to_lo": item_relevance_report,
            "item_faithfulness": item_faithfulness_report,
            "item_answerability": item_answerability_report,
            "item_processing_time": timing_report,
            "question_diagnostics": diagnostics_report,
        },
    )


def _renumber_rows(report: dict, new_ids: dict) -> dict:
    for row in report.get("items", []):
        if row.get("item_id") in new_ids:
            row["item_id"] = new_ids[row["item_id"]]
    report["items"].sort(key=lambda row: row.get("item_id") or 0)
    return report


def _count_stats(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "avg": 0.0, "max": 0}
    return {"min": min(values), "avg": round(sum(values) / len(values), 2), "max": max(values)}


def _types(items) -> dict:
    counts = Counter(item.get("type") for item in items)
    return {question_type: counts.get(question_type, 0) for question_type in QUESTION_TYPES}


def build_question_diagnostics(
    los, all_items, status, normalized_items, run: _QuestionRun, qs: QuestionSettings, *,
    under_minimum, coverage_calls, sections_filled, validation_report,
) -> dict:
    """Report question_diagnostics: dropped questions by reason, questions per section and
    type, extra rounds, duplicates, batch splits, the coverage call and the warnings."""
    by_reason = Counter({reason: 0 for reason in DROP_REASONS})
    by_reason.update(run.generation_dropped)
    accepted = [item for item in all_items if status.get(item["id"]) == STATUS_ACCEPTED]
    for item in all_items:
        reason = status.get(item["id"])
        if reason in DROP_REASONS:
            by_reason[reason] += 1
    by_reason[DROP_UNKNOWN_LO_ID] += len(accepted) - len(normalized_items)

    generated_per_section = Counter(item.get("lo_id") for item in all_items)
    accepted_per_section = Counter(item.get("lo_id") for item in normalized_items)
    section_ids = [lo.get("id") for lo in los]

    warnings = []
    for lo in los:
        lo_id = lo.get("id")
        if accepted_per_section[lo_id]:
            continue
        if generated_per_section[lo_id]:
            stage = "filter"
            reason = "all generated questions were dropped (validation, type rules, duplicates or score filter)"
        else:
            stage = "generation"
            reason = run.no_content.get(lo_id) or "the model returned no question, also not in the coverage call"
        bloom = lo.get("bloom_level", "")
        warnings.append({
            "code": WARNING_SECTION_WITHOUT_QUESTIONS,
            "message": f"Section {lo_id} (Bloom: {bloom}) has no question ({stage}): {reason}",
            "section_id": lo_id,
            "bloom_level": bloom,
            "stage": stage,
            "reason": reason,
        })
    for entry in run.wrong_marking:
        item = entry["item"]
        letters = ", ".join(entry["wrong_choices"]) or "?"
        warnings.append({
            "code": WARNING_CHOICE_MARKING,
            "message": f"Question {item['id']} (section {item.get('lo_id')}) dropped: option(s) {letters} marked wrongly",
            "section_id": item.get("lo_id"),
            "question_id": item["id"],
        })
    page_refs = sorted({w["item_id"] for w in validation_report.get("warning_items", [])
                        if w.get("code") == WARNING_PAGE_REFERENCE and w.get("item_id") is not None})
    if page_refs:
        warnings.append({
            "code": WARNING_PAGE_REFERENCE,
            "message": f"{len(page_refs)} question(s) refer to pages or chapters of the document: {page_refs}",
            "question_ids": page_refs,
        })
    low_share = []
    if "choice" in qs.types and len(qs.types) > 1:
        for lo_id in section_ids:
            section_items = [item for item in normalized_items if item.get("lo_id") == lo_id]
            if section_items:
                share = sum(1 for item in section_items if item.get("type") == "choice") / len(section_items)
                if share < run.settings.question_choice_min_share:
                    low_share.append(lo_id)
    if low_share:
        warnings.append({
            "code": WARNING_CHOICE_SHARE,
            "message": (f"{len(low_share)} section(s) have less than "
                        f"{int(round(run.settings.question_choice_min_share * 100))} % choice questions: {low_share}"),
            "section_ids": low_share,
        })

    origins = Counter(item.get("origin") for item in all_items)
    return {
        "per_section_minimum": qs.per_section,
        "question_options": {
            "types": list(qs.types), "per_section": qs.per_section,
            "choice_options": qs.choice_options, "hints": qs.hints,
        },
        "sections_total": len(los),
        "questions_generated": len(all_items),
        "questions_accepted": len(normalized_items),
        "questions_per_section": {
            "generated": _count_stats([generated_per_section[i] for i in section_ids]),
            "accepted": _count_stats([accepted_per_section[i] for i in section_ids]),
        },
        "question_types": {
            "generated": _types(all_items),
            "accepted": _types(normalized_items),
            "practical_accepted": sum(1 for item in normalized_items if item.get("practical")),
        },
        "questions_by_origin": dict(origins),
        "generation_calls": {
            "initial": run.calls.get("initial", 0),
            "extra": run.calls.get("extra", 0),
            "coverage": run.calls.get("coverage", 0),
        },
        "extra_rounds": {
            "configured": run.extra_rounds,
            "rounds_run": sum(1 for r in run.rounds if r["calls"]),
            "rounds": run.rounds,
        },
        "batch_splits": run.splits,
        "dedup_threshold": run.settings.question_dedup_threshold,
        "duplicates_removed": {
            "total": sum(run.duplicates_by_origin.values()),
            "by_origin": dict(run.duplicates_by_origin),
        },
        "wrong_choice_marking": len(run.wrong_marking),
        "times_clamped": run.times_clamped,
        "validation_warnings": dict(validation_report.get("warning_counts", {})),
        "supplementary_call": {
            "sections_missing": list(under_minimum),
            "calls": coverage_calls,
            "sections_filled": list(sections_filled),
        },
        "sections_without_content": {str(k): v for k, v in run.no_content.items()},
        "items_dropped": {
            "total": sum(by_reason.values()),
            "by_reason": dict(by_reason),
        },
        "warnings": warnings,
    }


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
    evaluation = item.get("evaluation", {})
    score = evaluation.get("score") if isinstance(evaluation, dict) else None
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
