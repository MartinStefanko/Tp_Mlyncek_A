"""Whole run: POST /v1/jobs → RQ SimpleWorker → completed → /result and /report.

Also the error paths of run_job (llm_error, extraction_failed, internal_error, time limit)
and cancellation before and during the run.
"""

import json

import jsonschema
import pytest

from qgen.contract import Report, UnitsResult
from qgen.contract.schemas import build_schema
from qgen.extraction.text_extraction import OcrUnavailableError, pdfs_to_text
from qgen.progress import PIPELINE_STAGES
from tests.baseline_data import baseline_models, load_json, replay_client, require_baseline
from tests.fakes import FakeLLM, client_error
from tests.worker.conftest import EKO_PDF, HEADERS, SEGMENTS, fixed_extraction, scripted_answer

SK = {"external_ref": "gen-1", "language": "sk"}


def _result_and_report(env, job_id):
    result = env.client.get(f"/v1/jobs/{job_id}/result", headers=HEADERS)
    report = env.client.get(f"/v1/jobs/{job_id}/report", headers=HEADERS)
    assert result.status_code == 200, result.text
    assert report.status_code == 200, report.text
    jsonschema.Draft202012Validator(build_schema("lesson")).validate(result.json())
    jsonschema.Draft202012Validator(build_schema("report")).validate(report.json())
    return UnitsResult.model_validate(result.json()), Report.model_validate(report.json())


def test_post_worker_completed_result_and_report(env):
    job_id = env.create(SK)
    assert env.status(job_id)["status"] == "queued"
    env.work()

    status = env.status(job_id)
    assert (status["status"], status["stage"], status["progress"]) == ("completed", "output", 1.0)
    record = env.store.get(job_id)
    assert record.attempts == 1
    assert not env.store.has_inputs(job_id)  # PDFs deleted after completed

    units, report = _result_and_report(env, job_id)
    assert len(units.units) == 1
    questions = units.units[0].questions
    # 3 sections x (choice, choice, open); the types come straight from the pipeline
    assert len(questions) == 9 == len(report.question_meta)
    assert [q.type for q in questions[:3]] == ["choice", "choice", "open"]
    first = questions[0]
    assert (first.type, first.time_to_handle, first.time_to_explain) == ("choice", 60, 30)
    assert first.name == "Otázka 1"
    assert [c.is_correct for c in first.choices] == [True, False, False, False]
    # one hint -> name "Nápoveda"; <b> is not an allowed tag, nh3 removes it
    assert [h.model_dump() for h in first.hints] == [{"name": "Nápoveda", "url": "<p>Pozri definíciu.</p>"}]
    assert [h.name for h in questions[1].hints] == ["Ľahká nápoveda", "Ťažká nápoveda"]
    assert questions[2].choices[0].text == "Odpoveď 3/1."
    assert [s.id for s in report.sections] == ["S-1", "S-2", "S-3"]
    meta = report.question_meta[0]
    assert (meta.unit, meta.question, meta.section_id, meta.bloom_level) == (0, 0, "S-1", "understand")
    assert meta.source[0].file == "bio.pdf" and meta.source[0].pages == [1]
    assert report.quality.section_faithfulness_avg == 1.0
    assert {"extract", "learning_objects", "questions", "output"} <= set(report.timings)


def test_progress_is_written_to_job_json(env):
    seen = []
    original_update = env.store.update

    def recording_update(job_id, change):
        record = original_update(job_id, change)
        seen.append((record.status, record.stage, record.progress))
        return record

    env.store.update = recording_update
    job_id = env.create(SK)
    env.work()
    running = [(stage, progress) for status, stage, progress in seen if status == "running" and stage]
    stages = [stage for stage, _ in running]
    assert stages[0] == "extract"
    assert "questions" in stages and "q_evaluation" in stages
    order = [(*PIPELINE_STAGES, "output").index(stage) for stage in stages]
    assert order == sorted(order)  # the step only moves forward
    progresses = [progress for _, progress in running]
    assert progresses == sorted(progresses) and max(progresses) < 1.0
    assert env.status(job_id)["status"] == "completed"


def test_progress_writes_are_throttled(env):
    ticks = iter(range(0, 100000))
    env.use(progress_interval_s=1000, monotonic=lambda: float(next(ticks)))
    writes = []
    original_update = env.store.update

    def counting_update(job_id, change):
        record = original_update(job_id, change)
        writes.append(record.stage)
        return record

    env.store.update = counting_update
    env.create(SK)
    env.work()
    running_writes = writes[1:-1]  # without queued→running and →completed
    assert len(running_writes) == len(set(running_writes))  # one write per step


@pytest.mark.parametrize("ocr", [False, True])
def test_options_reach_the_pipeline(env, ocr):
    calls = {}

    def extract(inputs, ocr=None, progress_callback=None, settings=None):
        calls["inputs"], calls["ocr"] = inputs, ocr
        return fixed_extraction(SEGMENTS)(inputs, ocr, progress_callback, settings)

    def pipeline(segments, **kwargs):
        calls.update(kwargs)
        raise RuntimeError("stop here")

    env.use(extract=extract, pipeline=pipeline)
    job_id = env.create({"language": "en", "extraction": {"ocr": ocr}, "questions": {"per_section": 5}})
    env.work()
    assert calls["ocr"] is ocr
    assert calls["inputs"][0]["name"] == "bio.pdf" and calls["inputs"][0]["path"].endswith("D1.pdf")
    assert calls["language"] == "en"
    question_options = calls["question_options"]
    assert (question_options.per_section, question_options.types) == (5, ["choice", "open", "no_answer"])
    assert (question_options.choice_options, question_options.hints) == (4, 2)
    assert env.status(job_id)["status"] == "failed"


def test_replay_eko_completed(env_factory):
    """eko.pdf over baseline_v3: counts must match the recording."""
    require_baseline()
    env = env_factory(**baseline_models())
    replay = replay_client()
    env.use(llm=replay, embedder=replay, extract=fixed_extraction(load_json("segments.json")))
    job_id = env.create({"external_ref": "eko-1"}, pdf=EKO_PDF.read_bytes(), name="eko.pdf")
    env.work()
    assert env.status(job_id)["status"] == "completed", env.status(job_id)

    units, report = _result_and_report(env, job_id)
    assert len(units.units[0].questions) == len(load_json("questions.json"))
    assert len(report.sections) == len(load_json("learning_objects.json"))
    assert {m.source[0].file for m in report.question_meta if m.source} == {"eko.pdf"}
    assert all(q.type == "open" for q in units.units[0].questions)


# --- errors ---------------------------------------------------------------------------------


def test_llm_error_fails_the_job_and_keeps_the_pdfs(env):
    env.use(llm=FakeLLM(default=client_error(400)))
    job_id = env.create(SK)
    env.work()
    status = env.status(job_id)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "llm_error"
    assert status["retryable_until"] is not None
    assert env.store.has_inputs(job_id)

    # retry finds the PDFs and runs again
    env.use(llm=FakeLLM(default=scripted_answer))
    assert env.client.post(f"/v1/jobs/{job_id}/retry", headers=HEADERS).status_code == 202
    env.work()
    assert env.status(job_id)["status"] == "completed"


def test_pdf_without_text_is_extraction_failed(env):
    env.use(extract=pdfs_to_text)  # real extraction of a blank PDF
    job_id = env.create(SK)
    env.work()
    error = env.status(job_id)["error"]
    assert error["code"] == "extraction_failed"
    assert error["details"] == {"ocr": False}
    assert env.store.has_inputs(job_id)


def test_ocr_unavailable_is_extraction_failed(env):
    def extract(*args, **kwargs):
        raise OcrUnavailableError("easyocr missing")

    env.use(extract=extract)
    job_id = env.create(SK)
    env.work()
    assert env.status(job_id)["error"]["code"] == "extraction_failed"


def test_unexpected_error_is_internal_error_without_its_text(env):
    def pipeline(*args, **kwargs):
        raise KeyError("secret document text")

    env.use(pipeline=pipeline)
    job_id = env.create(SK)
    env.work()
    error = env.status(job_id)["error"]
    assert error["code"] == "internal_error"
    assert error["details"] == {"stage": "extract", "error_type": "KeyError"}
    assert "secret" not in json.dumps(error)


def test_time_limit_is_internal_error(env_factory):
    env = env_factory(job_timeout_s=5)
    ticks = iter(range(0, 100000, 10))  # every reading of the clock is 10 s later
    env.use(monotonic=lambda: float(next(ticks)))
    job_id = env.create(SK)
    env.work()
    error = env.status(job_id)["error"]
    assert error["code"] == "internal_error"
    assert error["details"]["reason"] == "timeout" and error["details"]["timeout_s"] == 5


# --- cancellation ---------------------------------------------------------------------------


def test_cancelled_job_in_queue_does_not_run(env):
    job_id = env.create(SK)
    assert env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS).status_code == 200
    env.work()
    record = env.store.get(job_id)
    assert (record.status, record.attempts) == ("cancelled", 0)


def test_cancel_during_the_run_stops_it(env):
    state = {"job_id": None, "cancelled": False}

    def answer(prompt, model):
        if not state["cancelled"]:
            state["cancelled"] = True
            response = env.client.delete(f"/v1/jobs/{state['job_id']}", headers=HEADERS)
            assert response.json()["status"] == "cancelled"
        return scripted_answer(prompt, model)

    llm = FakeLLM(default=answer)
    env.use(llm=llm)
    state["job_id"] = job_id = env.create(SK)
    env.work()
    status = env.status(job_id)
    assert status["status"] == "cancelled"
    assert not env.store.has_inputs(job_id)
    assert env.client.get(f"/v1/jobs/{job_id}/result", headers=HEADERS).json()["error"]["code"] == "job_not_completed"
    assert not any("LO 1:" in prompt for _, prompt in llm.calls)  # stopped before questions


def test_cancel_after_the_last_progress_is_a_normal_end(env):
    """Cancelled between the output and running → completed: InvalidTransition, normal end."""
    original_save_report = env.store.save_report

    def save_report_then_cancel(job_id, report):
        original_save_report(job_id, report)
        assert env.client.delete(f"/v1/jobs/{job_id}", headers=HEADERS).json()["status"] == "cancelled"

    env.store.save_report = save_report_then_cancel
    job_id = env.create(SK)
    env.work()
    assert env.status(job_id)["status"] == "cancelled"
    assert not env.store.has_inputs(job_id)
