"""Replay provider (QGEN_LLM_PROVIDER / QGEN_EMBED_PROVIDER = replay) for the service in development.

Small hand-written recordings only; no real LLM / embedding API.
"""

import json
import logging

import pytest

from qgen.config import Settings
from qgen.llm import factory, retry
from qgen.llm.adapters.replay import (
    EMBED,
    GENERATE,
    ReplayLLM,
    is_question_prompt,
)
from qgen.llm.errors import LLMConfigError, LLMError, ProviderError
from qgen.lo.generation import build_lo_generation_prompt
from qgen.questions.pipeline import build_item_evaluation_prompt, build_item_generation_prompt

MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def _write(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def recording(tmp_path):
    return _write(tmp_path / "llm_calls.jsonl", [
        {"seq": 1, "method": GENERATE, "model": MODEL, "contents": "p1", "output": {"text": "first"}},
        {"seq": 2, "method": GENERATE, "model": MODEL, "contents": "p1", "output": {"text": "second"}},
        {"seq": 3, "method": EMBED, "model": EMBED_MODEL, "contents": ["a", "b"],
         "output": {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}},
        {"seq": 4, "method": GENERATE, "model": MODEL, "contents": "blocked", "output": {"text_error": "x"}},
    ])


@pytest.fixture(autouse=True)
def clean_factory(monkeypatch):
    factory.clear_cache()
    monkeypatch.setattr(retry, "_sleep", lambda s: None)
    yield
    factory.clear_cache()


def _settings(recording, **values):
    return Settings(
        _env_file=None, llm_provider="replay", embed_provider="replay", llm_replay_path=recording, **values
    )


# --- ReplayLLM --------------------------------------------------------------------------------


def test_answers_by_key_and_repeats_answers_in_recorded_order(recording):
    replay = ReplayLLM(recording)
    texts = [replay.generate("p1", model=MODEL).text for _ in range(3)]
    assert texts == ["first", "second", "first"]  # not used up: a second job gets them again
    assert replay.embed(["a", "b"], model=EMBED_MODEL) == [[1.0, 0.0], [0.0, 1.0]]
    assert replay.generate("blocked", model=MODEL).text == ""


def test_unrecorded_call_is_llm_error_without_the_prompt(recording):
    settings = _settings(recording)
    replay = ReplayLLM(recording)
    with pytest.raises(LLMError) as caught:
        retry.generate_text(replay, "secret document text", role="lo", settings=settings)
    assert caught.value.attempts == 1  # not retried
    assert "secret" not in str(caught.value)
    with pytest.raises(LLMError):
        retry.embed_texts(replay, ["other"], settings=settings)
    with pytest.raises(ProviderError):  # another model is another key
        replay.generate("p1", model="gemini-3.5-flash")


def test_delay_before_every_call(recording):
    waits = []
    replay = ReplayLLM(recording, delay_s=1.5, sleep=waits.append)
    replay.generate("p1", model=MODEL)
    replay.embed(["a", "b"], model=EMBED_MODEL)
    assert waits == [1.5, 1.5]


@pytest.mark.parametrize("language", ["sk", "en"])
def test_question_prompt_markers_match_the_pipeline(language):
    assert is_question_prompt(build_item_generation_prompt("LO 1: x", language))
    assert not is_question_prompt(build_item_evaluation_prompt("Q1: x", language))
    assert not is_question_prompt(build_lo_generation_prompt("Strana 1\ntext", language))


def test_fail_switch_fails_only_question_generation(tmp_path):
    question_prompt = build_item_generation_prompt("LO 1: x", "sk")
    path = _write(tmp_path / "calls.jsonl", [
        {"seq": 1, "method": GENERATE, "model": MODEL, "contents": "p1", "output": {"text": "ok"}},
        {"seq": 2, "method": GENERATE, "model": MODEL, "contents": question_prompt, "output": {"text": "[]"}},
    ])
    settings = _settings(path, llm_replay_fail=True)
    replay = ReplayLLM(path, fail_questions=True)
    assert replay.generate("p1", model=MODEL).text == "ok"
    with pytest.raises(LLMError) as caught:
        retry.generate_text(replay, question_prompt, role="questions", settings=settings)
    assert caught.value.attempts == 1
    assert ReplayLLM(path).generate(question_prompt, model=MODEL).text == "[]"  # switch off


# --- configuration -------------------------------------------------------------------------------


def test_replay_from_the_environment(recording, monkeypatch):
    monkeypatch.setenv("QGEN_LLM_PROVIDER", "replay")
    monkeypatch.setenv("QGEN_EMBED_PROVIDER", "replay")
    monkeypatch.setenv("QGEN_LLM_REPLAY_PATH", str(recording))
    monkeypatch.setenv("QGEN_LLM_REPLAY_DELAY_S", "0.5")
    monkeypatch.setenv("QGEN_LLM_REPLAY_FAIL", "1")
    settings = Settings(_env_file=None)
    assert (settings.llm_provider, settings.embed_provider) == ("replay", "replay")
    assert settings.llm_replay_path == recording
    assert (settings.llm_replay_delay_s, settings.llm_replay_fail) == (0.5, True)


def test_factory_builds_one_client_for_both_roles(recording):
    settings = _settings(recording, llm_replay_delay_s=0.25)
    llm = factory.get_llm("questions", settings)
    assert isinstance(llm, ReplayLLM)
    assert factory.get_embedder(settings) is llm
    assert llm.delay_s == 0.25 and llm.fail_questions is False


def test_replay_needs_an_existing_recording(tmp_path):
    with pytest.raises(LLMConfigError, match="QGEN_LLM_REPLAY_PATH"):
        factory.check_config(Settings(_env_file=None, llm_provider="replay", embed_provider="replay"))
    with pytest.raises(LLMConfigError, match="does not exist"):
        factory.check_config(_settings(tmp_path / "missing.jsonl"))


def test_check_config_never_refuses_replay_but_warns_loudly(recording, caplog):
    caplog.set_level(logging.WARNING, logger="qgen.llm")
    factory.check_config(_settings(recording))  # gemini-* models are fine for a Gemini recording
    warning = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "REPLAY" in warning and "NEVER in production" in warning
    assert "QGEN_LLM_PROVIDER" in warning and "QGEN_EMBED_PROVIDER" in warning


def test_replay_only_for_embeddings_warns_too(recording, caplog):
    caplog.set_level(logging.WARNING, logger="qgen.llm")
    settings = Settings(
        _env_file=None, gemini_api_key="k", embed_provider="replay", llm_replay_path=recording
    )
    factory.check_config(settings, llm=False)
    assert any("QGEN_EMBED_PROVIDER=replay" in r.getMessage() for r in caplog.records)


def test_no_warning_without_replay(caplog):
    caplog.set_level(logging.WARNING, logger="qgen.llm")
    assert factory.warn_replay(Settings(_env_file=None, gemini_api_key="k")) is False
    assert not caplog.records
