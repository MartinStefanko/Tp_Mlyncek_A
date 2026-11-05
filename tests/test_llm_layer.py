"""Step 6: LLM interface, retry layer and the end of silent LLM errors.

No test calls a real API: FakeLLM / FakeEmbedder (tests/fakes.py) script the answers and
the errors; the Gemini adapter is tested with a fake SDK client object.
"""

import json
from types import SimpleNamespace

import pytest

from qgen.config import Settings
from qgen.llm import factory, retry
from qgen.llm.base import EmbeddingClient, LLMClient, LLMResponse
from qgen.llm.errors import LLMError, LLMTimeoutError, ProviderError, RateLimitError
from qgen.pipeline import run_pipeline
from tests.fakes import (
    INVALID_JSON,
    FakeEmbedder,
    FakeLLM,
    client_error,
    rate_limited,
    server_error,
    timed_out,
)
from qgen.llm.adapters.replay import ReplayClient

SEGMENTS = [
    {"page": 1, "source_id": "D1", "source_name": "a.pdf", "text": "Fotosyntéza je proces v rastlinách."},
    {"page": 2, "source_id": "D1", "source_name": "a.pdf", "text": "Dýchanie rastlín uvoľňuje energiu."},
]
TWO_LO = json.dumps([
    {"vzdelávací_objekt": "Fotosyntéza", "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
     "odporúčané_zadania": "Vysvetli fotosyntézu.", "citovane_zdroje": ["D1:1"]},
    {"vzdelávací_objekt": "Dýchanie", "bloom_level": "Pochopiť", "odporúčané_aktivity": ["čítanie"],
     "odporúčané_zadania": "Opíš dýchanie.", "citovane_zdroje": ["D1:2"]},
], ensure_ascii=False)


@pytest.fixture
def sleeps(monkeypatch):
    """Records the waits of the retry layer instead of sleeping."""
    waits = []
    monkeypatch.setattr(retry, "_sleep", waits.append)
    return waits


@pytest.fixture
def settings():
    return Settings.model_construct()  # code defaults: 5 attempts, 3 s delay, batch 100


def _json(llm, settings, role="lo"):
    return retry.generate_json(llm, "prompt", role=role, model="m", settings=settings)


# --- interface ------------------------------------------------------------------------

def test_fakes_and_replay_implement_the_interface(tmp_path):
    recording = tmp_path / "calls.jsonl"
    recording.write_text("", encoding="utf-8")
    replay = ReplayClient(recording)
    assert isinstance(FakeLLM(), LLMClient)
    assert isinstance(FakeEmbedder(), EmbeddingClient)
    assert isinstance(replay, LLMClient) and isinstance(replay, EmbeddingClient)


def test_replay_answers_by_method_model_and_input(tmp_path):
    recording = tmp_path / "calls.jsonl"
    lines = [
        {"seq": 1, "method": "generate_content", "model": "m", "contents": "p", "output": {"text": "[1]"}},
        {"seq": 2, "method": "embed_content", "model": "e", "contents": ["a", "b"],
         "output": {"embeddings": [[1.0], [2.0]]}},
        {"seq": 3, "method": "generate_content", "model": "m", "contents": "blocked",
         "output": {"text": {"text_error": "ValueError: blocked"}}},
    ]
    recording.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    replay = ReplayClient(recording)
    assert replay.embed(["a", "b"], model="e") == [[1.0], [2.0]]
    assert replay.generate("p", model="m") == LLMResponse(text="[1]", model="m", usage={})
    assert replay.generate("blocked", model="m").text == ""
    assert replay.calls == [("embed_content", "e"), ("generate_content", "m"), ("generate_content", "m")]


# --- retries --------------------------------------------------------------------------

def test_success_needs_one_call(settings, sleeps):
    llm = FakeLLM(['{"a": 1}'])
    assert _json(llm, settings) == {"a": 1}
    assert len(llm.calls) == 1 and sleeps == []


def test_429_with_retry_after_waits_and_succeeds(settings, sleeps):
    llm = FakeLLM([rate_limited(retry_after=7), "[1]"])
    assert _json(llm, settings) == [1]
    assert sleeps == [7]
    assert len(llm.calls) == 2


def test_429_without_retry_after_waits_exponentially(settings, sleeps):
    llm = FakeLLM([rate_limited(), rate_limited(), rate_limited(), "[1]"])
    assert _json(llm, settings) == [1]
    assert sleeps == [3.0, 6.0, 12.0]


@pytest.mark.parametrize("error", [server_error(500), server_error(503), ConnectionError("reset"), TimeoutError()])
def test_5xx_and_network_errors_are_retried(settings, sleeps, error):
    llm = FakeLLM([error, "[1]"])
    assert _json(llm, settings) == [1]
    assert sleeps == [3.0]


def test_timeout_is_a_failed_attempt(settings, sleeps):
    llm = FakeLLM([timed_out(), "[1]"])
    assert _json(llm, settings) == [1]
    assert sleeps == [3.0]


def test_timeout_on_every_attempt_raises_llm_error(settings, sleeps):
    llm = FakeLLM(default=timed_out())
    with pytest.raises(LLMError) as info:
        _json(llm, settings, role="check")
    assert isinstance(info.value.cause, LLMTimeoutError)
    assert info.value.attempts == 5 and len(llm.calls) == 5
    assert info.value.role == "check" and info.value.model == "m"
    assert not info.value.invalid_json
    assert len(sleeps) == 4


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_not_retried(settings, sleeps, status):
    error = client_error(status)
    llm = FakeLLM(default=error)
    with pytest.raises(LLMError) as info:
        _json(llm, settings)
    assert info.value.cause is error
    assert info.value.__cause__ is error
    assert info.value.attempts == 1 and len(llm.calls) == 1
    assert sleeps == []


def test_unknown_exception_is_not_retried(settings, sleeps):
    llm = FakeLLM(default=KeyError("bug"))
    with pytest.raises(LLMError) as info:
        _json(llm, settings)
    assert isinstance(info.value.cause, KeyError)
    assert len(llm.calls) == 1 and sleeps == []


def test_retries_come_from_config(sleeps):
    settings = Settings.model_construct(llm_retries=2, llm_retry_delay_s=0.5)
    llm = FakeLLM(default=server_error())
    with pytest.raises(LLMError):
        _json(llm, settings)
    assert len(llm.calls) == 2 and sleeps == [0.5]


def test_invalid_json_is_repeated_once(settings, sleeps):
    llm = FakeLLM([INVALID_JSON, '[{"id": 1}]'])
    assert _json(llm, settings) == [{"id": 1}]
    assert len(llm.calls) == 2


def test_invalid_json_twice_raises_llm_error(settings, sleeps):
    llm = FakeLLM([INVALID_JSON, ""])
    with pytest.raises(LLMError) as info:
        _json(llm, settings, role="questions")
    assert info.value.invalid_json is True
    assert info.value.role == "questions"
    assert len(llm.calls) == 2


def test_llm_error_message_has_no_prompt_or_answer(settings, sleeps):
    llm = FakeLLM([INVALID_JSON, INVALID_JSON])
    with pytest.raises(LLMError) as info:
        retry.generate_json(llm, "SECRET DOCUMENT TEXT", role="lo", model="m", settings=settings)
    assert "SECRET" not in str(info.value)
    assert INVALID_JSON not in str(info.value)


def test_safe_load_json_raises_json_decode_error():
    from qgen.llm.json_load import safe_load_json

    with pytest.raises(json.JSONDecodeError):
        safe_load_json("no json here")
    assert safe_load_json('text before [1, 2] after') == [1, 2]


# --- embeddings -----------------------------------------------------------------------

def test_embeddings_are_split_by_config(sleeps):
    settings = Settings.model_construct(embed_max_batch=2)
    embedder = FakeEmbedder()
    vectors = retry.embed_texts(embedder, ["a", "b", "c", "d", "e"], model="e", settings=settings)
    assert len(vectors) == 5
    assert embedder.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_embedder_failure_raises_llm_error(settings, sleeps):
    embedder = FakeEmbedder(default=server_error())
    with pytest.raises(LLMError) as info:
        retry.embed_texts(embedder, ["a"], model="e", settings=settings)
    assert info.value.role == "embed" and info.value.attempts == 5


def test_embedder_wrong_count_raises_llm_error(settings, sleeps):
    embedder = FakeEmbedder([[[1.0]]])
    with pytest.raises(LLMError):
        retry.embed_texts(embedder, ["a", "b"], model="e", settings=settings)


# --- pipeline: LLM errors are not hidden ----------------------------------------------

def test_run_pipeline_raises_when_llm_always_fails(settings, sleeps):
    llm = FakeLLM(default=server_error())
    with pytest.raises(LLMError):
        run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(), settings=settings)


def test_language_detection_failure_is_not_replaced_by_heuristic(settings, sleeps):
    llm = FakeLLM([INVALID_JSON, INVALID_JSON])
    with pytest.raises(LLMError) as info:
        run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(), language=None, settings=settings)
    assert info.value.invalid_json
    assert len(llm.calls) == 2  # only the detection call and its repetition


def test_invalid_json_in_lo_generation_fails_the_run(settings, sleeps):
    llm = FakeLLM(default=INVALID_JSON)
    with pytest.raises(LLMError) as info:
        run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(), language="sk", settings=settings)
    assert info.value.invalid_json and info.value.role == "lo"


def test_embedder_failure_fails_the_run(settings, sleeps):
    llm = FakeLLM(default=TWO_LO)  # two LO with the same Bloom level -> clustering embeds them
    embedder = FakeEmbedder(default=server_error())
    with pytest.raises(LLMError) as info:
        run_pipeline(SEGMENTS, llm=llm, embedder=embedder, language="sk", settings=settings)
    assert info.value.role == "embed"


def test_run_pipeline_with_fakes_succeeds(settings, sleeps):
    """Sanity check: with valid answers the fakes drive the whole pipeline without errors."""
    llm = FakeLLM(default=TWO_LO)
    result = run_pipeline(SEGMENTS, llm=llm, embedder=FakeEmbedder(), language="sk", settings=settings)
    assert len(result.all_learning_objects) >= 1
    assert sleeps == []


# --- factory --------------------------------------------------------------------------

def test_role_models_match_the_former_models(settings):
    assert factory.model_for_role("lo", settings) == settings.llm_model_lo
    assert factory.model_for_role("prereq", settings) == settings.llm_model_lo
    assert factory.model_for_role("questions", settings) == settings.llm_model_questions
    assert factory.model_for_role("eval", settings) == settings.llm_model_eval
    assert factory.model_for_role("check", settings) == settings.llm_model
    assert factory.model_for_role("topics", settings) == settings.llm_model
    assert factory.model_for_role("split", settings) == settings.llm_model
    with pytest.raises(ValueError):
        factory.model_for_role("unknown", settings)


def test_factory_rejects_missing_key():
    """Step 7: every provider exists; a missing key / base_url is a configuration error
    (LLMConfigError, a ValueError). More in tests/test_llm_providers.py."""
    with pytest.raises(ValueError):
        factory.get_llm("lo", Settings.model_construct())
    with pytest.raises(ValueError):
        factory.get_llm("lo", Settings.model_construct(llm_provider="openai_compatible", llm_model="m"))
    with pytest.raises(ValueError):
        factory.get_embedder(Settings.model_construct(embed_provider="openai_compatible", embed_model="e"))


def test_missing_llm_is_created_by_the_factory(settings, sleeps, monkeypatch):
    llm = FakeLLM(["[1]"])
    monkeypatch.setattr(factory, "get_llm", lambda role, s=None: llm)
    assert retry.generate_json(None, "p", role="topics", settings=settings) == [1]
    assert llm.calls == [(settings.llm_model, "p")]


# --- Gemini adapter (fake SDK client, no network) -------------------------------------

class _FakeModels:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def generate_content(self, **kwargs):
        self.calls.append(("generate_content", kwargs))
        if self.error:
            raise self.error
        return self.response

    def embed_content(self, **kwargs):
        self.calls.append(("embed_content", kwargs))
        if self.error:
            raise self.error
        return self.response


def _gemini():
    from qgen.llm.adapters import gemini

    return gemini


def test_gemini_calls_the_api_like_before():
    gemini = _gemini()
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    models = _FakeModels(SimpleNamespace(text="[1]", usage_metadata=usage))
    llm = gemini.GeminiLLM(client=SimpleNamespace(models=models))
    response = llm.generate("prompt", model="gemini-x", json_mode=True, temperature=0.9)
    assert models.calls == [("generate_content", {"model": "gemini-x", "contents": "prompt"})]
    assert response == LLMResponse("[1]", "gemini-x", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    emb_models = _FakeModels(SimpleNamespace(embeddings=[SimpleNamespace(values=[1, 2]), SimpleNamespace(values=[3, 4])]))
    embedder = gemini.GeminiEmbedder(client=SimpleNamespace(models=emb_models))
    assert embedder.embed(["a", "b"], model="emb") == [[1, 2], [3, 4]]
    assert emb_models.calls == [("embed_content", {"model": "emb", "contents": ["a", "b"]})]


def test_gemini_empty_answer_is_empty_text():
    gemini = _gemini()
    llm = gemini.GeminiLLM(client=SimpleNamespace(models=_FakeModels(SimpleNamespace(text=None))))
    assert llm.generate("p", model="m").text == ""


def test_gemini_timeout_is_set_in_milliseconds():
    gemini = _gemini()
    client = gemini.build_client("fake-key", 120)
    assert client._api_client._http_options.timeout == 120_000


def _api_error(code, headers=None, details=None):
    gemini = _gemini()
    response = SimpleNamespace(headers=headers or {})
    return gemini.genai_errors.APIError(code, details or {"error": {"code": code, "status": "X"}}, response)


def test_gemini_translates_http_errors():
    gemini = _gemini()
    limited = gemini.translate_error(_api_error(429, headers={"retry-after": "9"}))
    assert isinstance(limited, RateLimitError) and limited.retry_after == 9
    body = {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}
    assert gemini.translate_error(_api_error(429, details=body)).retry_after == 17
    assert gemini.translate_error(_api_error(429)).retry_after is None
    for code in (400, 401, 403, 404):
        error = gemini.translate_error(_api_error(code))
        assert isinstance(error, ProviderError) and not error.retryable and error.status == code
    for code in (408, 500, 503):
        assert gemini.translate_error(_api_error(code)).retryable


def test_gemini_translates_timeout_and_network_errors():
    import httpx

    gemini = _gemini()
    assert isinstance(gemini.translate_error(httpx.ReadTimeout("slow")), LLMTimeoutError)
    network = gemini.translate_error(httpx.ConnectError("down"))
    assert isinstance(network, ProviderError) and network.retryable
    other = ValueError("x")
    assert gemini.translate_error(other) is other


def test_gemini_adapter_raises_translated_error():
    gemini = _gemini()
    models = _FakeModels(error=_api_error(403))
    llm = gemini.GeminiLLM(client=SimpleNamespace(models=models))
    with pytest.raises(ProviderError) as info:
        llm.generate("p", model="m")
    assert info.value.status == 403 and not info.value.retryable
    assert isinstance(info.value.__cause__, gemini.genai_errors.APIError)
