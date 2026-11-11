"""Parallel LLM calls (krok 13b): qgen.llm.parallel.run_batches and the pipeline with
QGEN_LLM_MAX_CONCURRENCY > 1.

- results in the order of the batches, whatever finishes first; same pipeline result as
  with concurrency 1 (FakeLLM with random-like delays, and replay of baseline_v4)
- never more calls at once than the limit (all batches and steps together)
- LLMError in one batch: no new batch starts, the running ones finish, no partial result
- cancel (JobCancelled from the progress callback) during a parallel step
- progress only from the main thread and monotonic
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import pytest

from qgen.config import Settings
from qgen.llm import retry
from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMError
from qgen.llm.parallel import run_batches
from qgen.pipeline import run_pipeline
from service.jobs.runner import JobCancelled
from tests.baseline_data import baseline_settings, load_json, replay_client, require_baseline
from tests.fakes import (
    EMPTY_ITEMS,
    OrthogonalVectors,
    client_error,
    is_coverage_call,
    is_extra_round,
    is_question_generation,
    items,
    question,
    section_ids,
)
from tests.test_progress import Recorder, assert_well_formed


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda seconds: None)


def settings_with(limit: int, **fields) -> Settings:
    return Settings.model_construct(llm_max_concurrency=limit, **fields)


# --- run_batches -------------------------------------------------------------------------


def test_results_keep_the_order_of_the_tasks_when_later_ones_finish_first():
    def task(index):
        def run():
            time.sleep(0.05 * (4 - index))  # the last task finishes first
            return index
        return run

    finished = []
    results = run_batches([task(i) for i in range(5)], settings=settings_with(5),
                          on_done=lambda index, result: finished.append(index))
    assert results == [0, 1, 2, 3, 4]
    assert finished[0] == 4 and sorted(finished) == [0, 1, 2, 3, 4]


def test_limit_one_runs_in_the_calling_thread_one_after_another():
    threads, order = [], []

    def task(index):
        def run():
            threads.append(threading.current_thread())
            order.append(index)
            return index
        return run

    assert run_batches([task(i) for i in range(4)], settings=settings_with(1)) == [0, 1, 2, 3]
    assert order == [0, 1, 2, 3]
    assert all(thread is threading.current_thread() for thread in threads)


def test_at_most_limit_tasks_run_at_once_and_on_done_runs_in_the_main_thread():
    lock, running, peak, callback_threads = threading.Lock(), [0], [0], []

    def task():
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1
        return True

    run_batches([task] * 12, settings=settings_with(3),
                on_done=lambda index, result: callback_threads.append(threading.current_thread()))
    assert peak[0] == 3
    assert callback_threads and all(thread is threading.main_thread() for thread in callback_threads)


def test_failure_stops_new_tasks_lets_running_ones_finish_and_returns_nothing():
    started, finished = [], []

    def task(index):
        def run():
            started.append(index)
            if index == 1:
                raise LLMError("boom", role="questions", model="m", attempts=1)
            time.sleep(0.1)
            finished.append(index)
            return index
        return run

    with pytest.raises(LLMError):
        run_batches([task(i) for i in range(6)], settings=settings_with(2))
    assert sorted(started) == [0, 1]  # nothing new after the failure
    assert finished == [0]            # the running task finished


def test_the_error_of_the_lowest_failed_batch_is_raised():
    gate = threading.Barrier(2)

    def failing(index):
        def run():
            gate.wait(timeout=5)  # both fail at about the same time
            if index == 1:
                time.sleep(0.05)  # the lower batch fails later
            raise ValueError(f"batch {index}")
        return run

    with pytest.raises(ValueError, match="batch 1"):
        run_batches([lambda: 0, failing(1), failing(2)], settings=settings_with(3))


def test_exception_from_on_done_stops_new_tasks():
    started = []

    def task(index):
        def run():
            started.append(index)
            time.sleep(0.02 if index == 0 else 0.1)
            return index
        return run

    def on_done(index, result):
        raise JobCancelled()

    with pytest.raises(JobCancelled):
        run_batches([task(i) for i in range(6)], settings=settings_with(2), on_done=on_done)
    assert sorted(started) == [0, 1]


# --- the pipeline with a fake model ----------------------------------------------------------

PAGES = 18
SEGMENTS = [
    {"page": page, "source_id": "D1", "source_name": "a.pdf", "text": f"Strana {page}: pojem číslo {page} a jeho vlastnosti."}
    for page in range(1, PAGES + 1)
]
PIPELINE_FIELDS = dict(
    lo_batch_size=3,                     # 6 LO generation batches
    question_generation_batch_size=2,
    question_eval_batch_size=4,
    question_extra_rounds=1,
)


def _delay(prompt: str) -> float:
    """Random-like but fixed delay per prompt, so batches finish in a mixed order."""
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:2], 16) % 6 * 0.004


def _question_round(prompt: str) -> int:
    if is_coverage_call(prompt):
        return 4
    return 3 if is_extra_round(prompt) else 1


def answer(prompt: str) -> str:
    """Deterministic answer of every prompt of the pipeline."""
    if "extrahuj merateľné vzdelávacie ciele" in prompt:
        pages = [int(p) for p in re.findall(r"strana (\d+)\]", prompt)]
        return items({"vzdelávací_objekt": f"Pojem {p}", "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
                      "odporúčané_zadania": "Vysvetli pojem.", "citovane_zdroje": [f"D1:{p}"]} for p in pages)
    if is_question_generation(prompt):
        n = _question_round(prompt)
        rows = []
        for lo_id in section_ids(prompt):
            rows.append(question(lo_id, n))
            if n == 1:
                rows.append(question(lo_id, 2, "open"))
        return items(rows)
    if "Item id:" in prompt:  # quality score
        ids = [int(x) for x in re.findall(r"Item id: (\d+)", prompt)]
        return items({"id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids)
    if "ITEM ID:" in prompt:  # faithfulness / answerability: every 4th question fails (depends on the ids)
        ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
        return items({"item_id": i, "skore": 2 if i % 4 == 0 else 5, "zdovodnenie": "ok", "choices_ok": True,
                      "wrong_choices": []} for i in ids)
    if "LO ID:" in prompt:
        ids = [int(x) for x in re.findall(r"LO ID: (\d+)", prompt)]
        return items({"lo_id": i, "skore": 5, "zdovodnenie": "ok"} for i in ids)
    return EMPTY_ITEMS


class CountingModel:
    """LLMClient + EmbeddingClient that measures the largest number of calls at once."""

    def __init__(self, fail_on=None, on_call=None):
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.prompts: list[str] = []
        self.vectors = OrthogonalVectors()
        self.fail_on = fail_on
        self.on_call = on_call

    def _enter(self):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)

    def _leave(self):
        with self.lock:
            self.running -= 1

    def generate(self, prompt, *, model, json_mode=True, temperature=0.2, max_tokens=None):
        self._enter()
        try:
            with self.lock:
                self.prompts.append(prompt)
            if self.on_call:
                self.on_call(prompt)
            time.sleep(_delay(prompt))
            if self.fail_on and self.fail_on(prompt):
                raise client_error(400)
            return LLMResponse(text=answer(prompt), model=model, usage={})
        finally:
            self._leave()

    def embed(self, texts, *, model):
        self._enter()
        try:
            time.sleep(0.002)
            with self.lock:
                return self.vectors(texts, model)
        finally:
            self._leave()

    def question_calls(self) -> list[str]:
        return [p for p in self.prompts if is_question_generation(p)]


def _run(limit, model=None, progress_callback=None):
    model = model or CountingModel()
    result = run_pipeline(SEGMENTS, llm=model, embedder=model, language="sk", progress_callback=progress_callback,
                          settings=settings_with(limit, **PIPELINE_FIELDS))
    return result, model


TIMING_REPORTS = ("lo_processing_time", "item_processing_time")


def _as_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _comparable(result):
    reports = {name: report for name, report in result.reports.items() if name not in TIMING_REPORTS}
    return result.learning_objects, result.all_learning_objects, result.questions, result.all_questions, reports


@pytest.fixture(scope="module")
def sequential_run():
    previous = retry._sleep
    retry._sleep = lambda seconds: None
    try:
        return _run(1)
    finally:
        retry._sleep = previous


def test_fake_pipeline_exercises_every_parallel_step(sequential_run):
    result, model = sequential_run
    diagnostics = result.reports["question_diagnostics"]
    assert model.peak == 1
    assert len(result.learning_objects) == PAGES
    assert diagnostics["generation_calls"]["extra"] > 0
    assert diagnostics["generation_calls"]["coverage"] > 0  # the dropped questions start coverage calls
    assert diagnostics["items_dropped"]["by_reason"]["low_score"] > 0


@pytest.mark.parametrize("limit", [2, 4])
def test_parallel_run_gives_the_same_result_as_the_sequential_one(sequential_run, limit):
    expected, _ = sequential_run
    result, model = _run(limit)
    assert model.peak > 1
    assert _comparable(result) == _comparable(expected)


@pytest.mark.parametrize("limit", [2, 3])
def test_calls_at_once_never_exceed_the_limit(limit):
    _, model = _run(limit)
    assert 1 < model.peak <= limit


def test_llm_error_in_one_batch_stops_the_step_without_a_partial_result():
    # The first question batch (sections 1, 2) fails at once; the second one is still running.
    started_after_failure = []
    failed = threading.Event()

    def fails(prompt):
        return is_question_generation(prompt) and section_ids(prompt)[:1] == [1]

    def on_call(prompt):
        if is_question_generation(prompt):
            if failed.is_set():
                started_after_failure.append(section_ids(prompt))
            if fails(prompt):
                failed.set()
            else:
                time.sleep(0.1)

    model = CountingModel(fail_on=fails, on_call=on_call)
    with pytest.raises(LLMError) as error:
        _run(2, model)
    assert error.value.role == "questions"
    assert len(model.question_calls()) == 2      # batch 1 (failed) and batch 2 (finished)
    assert started_after_failure == [[3, 4]]     # only the batch already running, no new one
    assert not any(is_extra_round(p) or "Item id:" in p for p in model.prompts)


def test_cancel_during_a_parallel_step_starts_no_new_batch():
    def progress(stage, done, total):
        if stage == "questions" and done >= 1:
            raise JobCancelled()

    model = CountingModel()
    with pytest.raises(JobCancelled):
        _run(2, model, progress_callback=progress)
    assert len(model.question_calls()) == 2  # the two batches that were running
    assert not any("Item id:" in p for p in model.prompts)


def test_progress_is_monotonic_and_only_from_the_main_thread():
    threads = []

    class ThreadRecorder(Recorder):
        def __call__(self, stage, done, total):
            threads.append(threading.current_thread())
            super().__call__(stage, done, total)

    recorder = ThreadRecorder()
    _run(4, progress_callback=recorder)
    assert all(thread is threading.main_thread() for thread in threads)
    assert recorder.stages() == ["language", "learning_objects", "lo_evaluation", "language", "questions",
                                 "q_evaluation"]
    assert_well_formed(recorder)
    units = dict(recorder.segments())["questions"]
    assert units[-1][0] == units[-1][1] == units[0][1]  # total stays batches * (1 + extra rounds)


# --- replay of baseline_v4 -----------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 4])
def test_replay_gives_the_recorded_result_with_and_without_parallel_calls(limit):
    require_baseline()
    client = replay_client()
    settings = baseline_settings().model_copy(update={"llm_max_concurrency": limit})
    result = run_pipeline(load_json("segments.json"), llm=client, embedder=client, settings=settings)
    assert _as_json(result.questions) == load_json("questions.json")
    assert _as_json(result.all_questions) == load_json("questions_all.json")
    assert client.unused_records() == []


def test_replay_reports_are_the_same_at_concurrency_one_and_four():
    require_baseline()
    results = []
    for limit in (1, 4):
        client = replay_client()
        settings = baseline_settings().model_copy(update={"llm_max_concurrency": limit})
        results.append(run_pipeline(load_json("segments.json"), llm=client, embedder=client, settings=settings))
    assert _comparable(results[0]) == _comparable(results[1])
