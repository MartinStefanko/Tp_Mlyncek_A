"""Step 7: OpenAI-compatible and Anthropic adapters, factory per provider, model inheritance
checks, start-up configuration check and the settings / import guards.

No network: the adapters run the real SDK over an httpx2.MockTransport (the HTTP request is
inspected and the response is scripted), or get a fake SDK client object.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest

from qgen.config import Settings
from qgen.llm import factory, retry
from qgen.llm.adapters import anthropic as anthropic_adapter
from qgen.llm.adapters import openai_compatible as openai_adapter
from qgen.llm.adapters._http import parse_retry_after
from qgen.llm.adapters.anthropic import AnthropicLLM
from qgen.llm.adapters.gemini import GeminiEmbedder, GeminiLLM
from qgen.llm.adapters.openai_compatible import OpenAICompatEmbedder, OpenAICompatLLM
from qgen.llm.base import EmbeddingClient, LLMClient, LLMResponse
from qgen.llm.errors import LLMConfigError, LLMError, LLMTimeoutError, ProviderError, RateLimitError
from tests.fakes import FakeLLM

REPO = Path(__file__).resolve().parent.parent
QGEN = REPO / "qgen"


@pytest.fixture(autouse=True)
def _fresh_factory_cache():
    factory.clear_cache()
    yield
    factory.clear_cache()


# --- HTTP mock helpers ----------------------------------------------------------------

class MockServer:
    """Records requests and answers them with scripted responses (or raises)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        step = self.responses.pop(0)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step("scripted", request=request)
        return step

    @property
    def body(self):
        return json.loads(self.requests[-1].content)

    def http_client(self):
        return httpx2.Client(transport=httpx2.MockTransport(self))


def _openai_sdk(server, base_url="http://llm:8001/v1", api_key=""):
    # The SDK module through the adapter: tests must not import vendor libraries either.
    return openai_adapter.openai.OpenAI(
        base_url=base_url,
        api_key=api_key or openai_adapter.PLACEHOLDER_API_KEY,
        max_retries=0,
        http_client=server.http_client(),
    )


def _anthropic_sdk(server):
    return anthropic_adapter.anthropic.Anthropic(api_key="test-key", max_retries=0, http_client=server.http_client())


CHAT_OK = httpx2.Response(200, json={
    "id": "c1", "object": "chat.completion", "created": 1, "model": "served-model",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "[1]"}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
})
EMBED_OK = httpx2.Response(200, json={
    "object": "list", "model": "emb",
    "data": [  # out of order on purpose: the adapter sorts by index
        {"object": "embedding", "index": 1, "embedding": [3.0, 4.0]},
        {"object": "embedding", "index": 0, "embedding": [1.0, 2.0]},
    ],
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
})
MESSAGE_OK = httpx2.Response(200, json={
    "id": "m1", "type": "message", "role": "assistant", "model": "claude-test",
    "content": [{"type": "text", "text": "[1, "}, {"type": "text", "text": "2]"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 9, "output_tokens": 4},
})


def _status(code, headers=None):
    return httpx2.Response(code, headers=headers or {}, json={"error": {"message": "scripted"}})


# --- OpenAI-compatible adapter --------------------------------------------------------

def test_openai_generate_request():
    server = MockServer(CHAT_OK)
    llm = OpenAICompatLLM(client=_openai_sdk(server))
    response = llm.generate("prompt text", model="qwen-27b", json_mode=True)
    request = server.requests[0]
    assert request.method == "POST" and str(request.url) == "http://llm:8001/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {openai_adapter.PLACEHOLDER_API_KEY}"
    assert server.body == {
        "model": "qwen-27b",
        "messages": [{"role": "user", "content": "prompt text"}],
        "temperature": 0.2,
    }
    assert response == LLMResponse("[1]", "served-model", {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14})


def test_openai_sends_max_tokens_and_temperature_when_given():
    server = MockServer(CHAT_OK)
    OpenAICompatLLM(client=_openai_sdk(server)).generate("p", model="m", temperature=0.7, max_tokens=500)
    assert server.body["temperature"] == 0.7 and server.body["max_tokens"] == 500


def test_openai_json_mode_is_not_sent_yet():
    """Decision of step 7: json_object allows only an object, the prompts ask for arrays."""
    assert openai_adapter.JSON_MODE_ENABLED is False
    server = MockServer(CHAT_OK, CHAT_OK)
    llm = OpenAICompatLLM(client=_openai_sdk(server))
    llm.generate("p", model="m", json_mode=True)
    llm.generate("p", model="m", json_mode=False)
    assert all("response_format" not in json.loads(r.content) for r in server.requests)


def test_openai_json_mode_maps_to_response_format_when_enabled(monkeypatch):
    monkeypatch.setattr(openai_adapter, "JSON_MODE_ENABLED", True)
    server = MockServer(CHAT_OK, CHAT_OK)
    llm = OpenAICompatLLM(client=_openai_sdk(server))
    llm.generate("p", model="m", json_mode=True)
    assert server.body["response_format"] == {"type": "json_object"}
    llm.generate("p", model="m", json_mode=False)
    assert "response_format" not in server.body


def test_openai_empty_answer_is_empty_text():
    empty = httpx2.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": None}}],
    })
    response = OpenAICompatLLM(client=_openai_sdk(MockServer(empty))).generate("p", model="m")
    assert response.text == "" and response.usage == {}


def test_openai_embed_request_and_order():
    server = MockServer(EMBED_OK)
    embedder = OpenAICompatEmbedder(client=_openai_sdk(server, base_url="http://emb:9000/v1", api_key="real-key"))
    assert embedder.embed(["a", "b"], model="emb") == [[1.0, 2.0], [3.0, 4.0]]
    request = server.requests[0]
    assert str(request.url) == "http://emb:9000/v1/embeddings"
    assert request.headers["authorization"] == "Bearer real-key"
    assert server.body == {"model": "emb", "input": ["a", "b"], "encoding_format": "float"}
    assert embedder.embed([], model="emb") == [] and len(server.requests) == 1


def test_openai_embed_wrong_count_is_an_error():
    with pytest.raises(ProviderError) as info:
        OpenAICompatEmbedder(client=_openai_sdk(MockServer(EMBED_OK))).embed(["only one"], model="emb")
    assert not info.value.retryable


def test_openai_build_client_settings():
    client = openai_adapter.build_client("http://llm:8001/v1", "", 120)
    assert client.api_key == openai_adapter.PLACEHOLDER_API_KEY  # empty key -> placeholder
    assert str(client.base_url).rstrip("/") == "http://llm:8001/v1"
    assert client.max_retries == 0 and client.timeout == 120
    assert openai_adapter.build_client("https://api.openai.com/v1", "sk-x", 30).api_key == "sk-x"
    with pytest.raises(LLMConfigError):
        openai_adapter.build_client("", "key", 30)


@pytest.mark.parametrize("call", ["generate", "embed"])
def test_openai_429_with_retry_after(call):
    server = MockServer(_status(429, {"retry-after": "7"}))
    client = _openai_sdk(server)
    with pytest.raises(RateLimitError) as info:
        if call == "generate":
            OpenAICompatLLM(client=client).generate("p", model="m")
        else:
            OpenAICompatEmbedder(client=client).embed(["a"], model="e")
    assert info.value.retry_after == 7 and info.value.retryable
    assert "prompt" not in str(info.value) and "scripted" not in str(info.value)  # no body in the message


def test_openai_429_retry_after_ms_wins():
    server = MockServer(_status(429, {"retry-after-ms": "1500", "retry-after": "9"}))
    with pytest.raises(RateLimitError) as info:
        OpenAICompatLLM(client=_openai_sdk(server)).generate("p", model="m")
    assert info.value.retry_after == 1.5


@pytest.mark.parametrize("code, retryable", [(400, False), (401, False), (403, False), (404, False),
                                             (408, True), (500, True), (503, True)])
def test_openai_http_errors(code, retryable):
    with pytest.raises(ProviderError) as info:
        OpenAICompatLLM(client=_openai_sdk(MockServer(_status(code)))).generate("p", model="m")
    assert info.value.status == code and info.value.retryable is retryable
    assert not isinstance(info.value, RateLimitError)


def test_openai_timeout_and_network_errors():
    with pytest.raises(LLMTimeoutError):
        OpenAICompatLLM(client=_openai_sdk(MockServer(httpx2.ReadTimeout))).generate("p", model="m")
    with pytest.raises(ProviderError) as info:
        OpenAICompatEmbedder(client=_openai_sdk(MockServer(httpx2.ConnectError))).embed(["a"], model="e")
    assert info.value.retryable and not isinstance(info.value, LLMTimeoutError)


def test_openai_translate_keeps_unknown_errors():
    other = KeyError("x")
    assert openai_adapter.translate_error(other) is other


# --- Anthropic adapter ----------------------------------------------------------------

def test_anthropic_generate_request():
    server = MockServer(MESSAGE_OK)
    llm = AnthropicLLM(client=_anthropic_sdk(server), max_tokens=4096)
    response = llm.generate("prompt text", model="claude-test", json_mode=True, temperature=0.9)
    request = server.requests[0]
    assert str(request.url).endswith("/v1/messages")
    # anthropic 2.x sends the key as a Bearer token; older versions used x-api-key.
    assert "test-key" in (request.headers.get("x-api-key") or request.headers.get("authorization", ""))
    assert server.body == {
        "model": "claude-test",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "prompt text"}],
        "system": anthropic_adapter.JSON_SYSTEM_INSTRUCTION,
    }
    assert "temperature" not in server.body  # not a parameter of the installed SDK
    assert response == LLMResponse("[1, 2]", "claude-test", {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13})


def test_anthropic_max_tokens_and_json_mode_off():
    server = MockServer(MESSAGE_OK, MESSAGE_OK)
    llm = AnthropicLLM(client=_anthropic_sdk(server))
    llm.generate("p", model="claude-test", json_mode=False)
    assert server.body["max_tokens"] == anthropic_adapter.DEFAULT_MAX_TOKENS == 8192
    assert "system" not in server.body
    llm.generate("p", model="claude-test", max_tokens=100)  # the caller's value wins
    assert server.body["max_tokens"] == 100


def test_anthropic_answer_without_text_is_empty():
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[], usage=None)))
    response = AnthropicLLM(client=fake).generate("p", model="claude-test")
    assert response.text == "" and response.model == "claude-test" and response.usage == {}


def test_anthropic_build_client_settings():
    client = anthropic_adapter.build_client("key", 45)
    assert client.max_retries == 0 and client.timeout == 45 and client.api_key == "key"
    with pytest.raises(LLMConfigError):
        anthropic_adapter.build_client("", 45)


@pytest.mark.parametrize("code, retryable", [(400, False), (401, False), (403, False), (404, False),
                                             (500, True), (529, True)])
def test_anthropic_http_errors(code, retryable):
    with pytest.raises(ProviderError) as info:
        AnthropicLLM(client=_anthropic_sdk(MockServer(_status(code)))).generate("p", model="claude-test")
    assert info.value.status == code and info.value.retryable is retryable


def test_anthropic_429_timeout_and_network():
    with pytest.raises(RateLimitError) as info:
        AnthropicLLM(client=_anthropic_sdk(MockServer(_status(429, {"retry-after": "12"})))).generate("p", model="c")
    assert info.value.retry_after == 12
    with pytest.raises(LLMTimeoutError):
        AnthropicLLM(client=_anthropic_sdk(MockServer(httpx2.ReadTimeout))).generate("p", model="c")
    with pytest.raises(ProviderError) as network:
        AnthropicLLM(client=_anthropic_sdk(MockServer(httpx2.ConnectError))).generate("p", model="c")
    assert network.value.retryable


def test_retry_layer_retries_a_translated_429(monkeypatch):
    """End to end: SDK 429 -> RateLimitError -> qgen.llm.retry waits Retry-After and repeats."""
    waits = []
    monkeypatch.setattr(retry, "_sleep", waits.append)
    server = MockServer(_status(429, {"retry-after": "4"}), CHAT_OK)
    llm = OpenAICompatLLM(client=_openai_sdk(server))
    assert retry.generate_json(llm, "p", role="check", model="m", settings=Settings.model_construct()) == [1]
    assert waits == [4.0] and len(server.requests) == 2


def test_retry_layer_does_not_retry_401(monkeypatch):
    monkeypatch.setattr(retry, "_sleep", lambda s: None)
    llm = AnthropicLLM(client=_anthropic_sdk(MockServer(_status(401))))
    with pytest.raises(LLMError) as info:
        retry.generate_text(llm, "p", role="lo", model="c", settings=Settings.model_construct())
    assert info.value.attempts == 1 and info.value.cause.status == 401


def test_parse_retry_after_forms():
    from datetime import datetime, timezone

    now = lambda: datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)  # noqa: E731
    assert parse_retry_after({"retry-after": "3"}) == 3
    assert parse_retry_after({"retry-after": "Wed, 04 Nov 2026 12:00:30 GMT"}, now=now) == 30
    assert parse_retry_after({"retry-after": "soon"}) is None
    assert parse_retry_after({}) is None and parse_retry_after(None) is None


# --- factory --------------------------------------------------------------------------

def _settings(**values):
    """Settings without .env. With another provider than gemini the eval model inherits
    QGEN_LLM_MODEL (empty QGEN_LLM_MODEL_EVAL), unless the test sets it: its code default
    gemini-3.5-flash would be rejected for that provider."""
    base = {"gemini_api_key": "g-key"}
    if values.get("llm_provider", "gemini") != "gemini":
        base["llm_model_eval"] = ""
    base.update(values)
    return Settings(_env_file=None, **base)


@pytest.fixture
def clean_env(monkeypatch):
    import os

    for name in list(os.environ):
        if name.upper().startswith("QGEN_") or name.upper() in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_factory_selects_adapter_by_provider(clean_env):
    assert isinstance(factory.get_llm("lo", _settings()), GeminiLLM)
    vllm = _settings(llm_provider="openai_compatible", llm_base_url="http://gpu-server:8001/v1", llm_model="qwen")
    assert isinstance(factory.get_llm("check", vllm), OpenAICompatLLM)
    claude = _settings(llm_provider="anthropic", anthropic_api_key="a-key", llm_model="claude-test")
    assert isinstance(factory.get_llm("eval", claude), AnthropicLLM)
    for client in (factory.get_llm("lo", vllm), factory.get_llm("lo", claude)):
        assert isinstance(client, LLMClient)


def test_factory_builds_one_client_per_provider_and_settings(clean_env):
    s = _settings(llm_provider="openai_compatible", llm_base_url="http://a/v1", llm_model="m")
    assert factory.get_llm("lo", s) is factory.get_llm("questions", s) is factory.get_llm("lo", s)
    other_url = _settings(llm_provider="openai_compatible", llm_base_url="http://b/v1", llm_model="m")
    assert factory.get_llm("lo", other_url) is not factory.get_llm("lo", s)
    assert factory.get_embedder(_settings()) is factory.get_embedder(_settings())


def test_factory_passes_settings_to_the_adapter(clean_env):
    s = _settings(llm_provider="anthropic", anthropic_api_key="a-key", llm_model="claude-test",
                  llm_max_tokens=2048, llm_timeout_s=33)
    llm = factory.get_llm("lo", s)
    assert llm._max_tokens == 2048 and llm._client.timeout == 33
    o = _settings(llm_provider="openai_compatible", llm_base_url="http://a/v1", llm_api_key="k", llm_model="m")
    client = factory.get_llm("lo", o)._client
    assert client.api_key == "k" and client.timeout == 120


def test_embedder_is_independent_of_the_llm_provider(clean_env):
    vllm_gemini = _settings(llm_provider="openai_compatible", llm_base_url="http://gpu-server:8001/v1",
                            llm_model="qwen", embed_provider="gemini")
    assert isinstance(factory.get_llm("lo", vllm_gemini), OpenAICompatLLM)
    assert isinstance(factory.get_embedder(vllm_gemini), GeminiEmbedder)
    claude_gemini = _settings(llm_provider="anthropic", anthropic_api_key="a", llm_model="claude-test")
    assert isinstance(factory.get_embedder(claude_gemini), GeminiEmbedder)
    gemini_local = _settings(embed_provider="openai_compatible", embed_base_url="http://llm:11434/v1",
                             embed_model="nomic-embed-text")
    assert isinstance(factory.get_llm("lo", gemini_local), GeminiLLM)
    embedder = factory.get_embedder(gemini_local)
    assert isinstance(embedder, OpenAICompatEmbedder) and isinstance(embedder, EmbeddingClient)
    assert str(embedder._client.base_url).startswith("http://llm:11434/v1")


def test_embed_api_key_falls_back_to_llm_api_key(clean_env):
    base = {"embed_provider": "openai_compatible", "embed_base_url": "http://e/v1", "embed_model": "e"}
    shared = _settings(llm_api_key="llm-key", **base)
    assert factory.get_embedder(shared)._client.api_key == "llm-key"
    own = _settings(llm_api_key="llm-key", embed_api_key="embed-key", **base)
    assert factory.get_embedder(own)._client.api_key == "embed-key"
    assert factory.get_embedder(own) is not factory.get_embedder(shared)  # key is part of the cache key
    # local vLLM without a key + cloud embeddings with their own key
    split = _settings(llm_provider="openai_compatible", llm_base_url="http://gpu-server:8001/v1",
                      llm_model="qwen", embed_api_key="embed-key", **base)
    assert factory.get_llm("lo", split)._client.api_key == openai_adapter.PLACEHOLDER_API_KEY
    assert factory.get_embedder(split)._client.api_key == "embed-key"
    assert _settings(**base).effective_embed_api_key == ""


def test_embed_api_key_from_env(clean_env):
    clean_env.setenv("QGEN_EMBED_API_KEY", "env-embed-key")
    s = Settings(_env_file=None)
    assert s.effective_embed_api_key == "env-embed-key"
    assert "env-embed-key" not in repr(s)


def test_anthropic_embeddings_are_rejected_at_start(clean_env):
    s = _settings(llm_provider="anthropic", anthropic_api_key="a", llm_model="claude-test",
                  embed_provider="anthropic")
    with pytest.raises(LLMConfigError, match="Anthropic has no embedding API"):
        factory.check_config(s)
    with pytest.raises(LLMConfigError, match="Anthropic has no embedding API"):
        factory.get_embedder(s)


@pytest.mark.parametrize("values, expected", [
    ({"gemini_api_key": ""}, "needs GEMINI_API_KEY"),
    ({"llm_provider": "openai_compatible", "llm_model": "m"}, "needs QGEN_LLM_BASE_URL"),
    ({"llm_provider": "anthropic", "llm_model": "c"}, "needs ANTHROPIC_API_KEY"),
    ({"embed_provider": "openai_compatible", "embed_model": "e"}, "needs QGEN_EMBED_BASE_URL"),
])
def test_missing_key_or_base_url_fails_at_start(clean_env, values, expected):
    s = _settings(**values)
    with pytest.raises(LLMConfigError, match=expected):
        factory.check_config(s)


def test_check_config_lists_every_problem(clean_env):
    s = Settings(_env_file=None, llm_provider="anthropic", embed_provider="openai_compatible")
    with pytest.raises(LLMConfigError) as info:
        factory.check_config(s)
    message = str(info.value)
    for part in ("ANTHROPIC_API_KEY", "QGEN_LLM_MODEL=", "QGEN_EMBED_BASE_URL", "QGEN_EMBED_MODEL=",
                 "QGEN_LLM_MODEL_EVAL=gemini-3.5-flash"):  # own default, not inherited
        assert part in message
    assert "QGEN_LLM_MODEL_LO" not in message  # inherited from QGEN_LLM_MODEL, reported there


def test_openai_compatible_key_may_be_empty(clean_env):
    factory.check_config(_settings(llm_provider="openai_compatible", llm_base_url="http://llm:8001/v1",
                                   llm_model="qwen"))


def test_check_config_skips_replaced_parts(clean_env):
    no_keys = Settings(_env_file=None)
    factory.check_config(no_keys, llm=False, embedder=False)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        factory.check_config(no_keys, llm=False)


def test_check_config_builds_the_clients(clean_env):
    s = _settings()
    factory.check_config(s)
    assert factory._llm_cache and factory._embedder_cache


def test_gemini_model_with_other_provider_is_rejected(clean_env):
    # The usual mistake: the provider changed, the default Gemini model stayed.
    with pytest.raises(LLMConfigError, match="QGEN_LLM_MODEL=gemini-3.1-flash-lite is a Gemini model"):
        factory.get_llm("lo", _settings(llm_provider="openai_compatible", llm_base_url="http://a/v1"))
    # A role model set to a Gemini model (as in .env.example) is named by its own variable.
    s = _settings(llm_provider="anthropic", anthropic_api_key="a", llm_model="claude-test",
                  llm_model_eval="gemini-3.5-flash")
    with pytest.raises(LLMConfigError) as info:
        factory.check_config(s)
    assert "QGEN_LLM_MODEL_EVAL=gemini-3.5-flash" in str(info.value)
    assert "QGEN_LLM_MODEL=" not in str(info.value)  # the base model is fine
    # Inherited role models are reported once, as QGEN_LLM_MODEL.
    s = _settings(llm_provider="openai_compatible", llm_base_url="http://a/v1", llm_model="gemini-x")
    with pytest.raises(LLMConfigError) as info:
        factory.check_config(s)
    assert str(info.value).count("is a Gemini model") == 1
    # With gemini everything is fine; another embed provider with a Gemini model is not.
    factory.check_config(_settings())
    with pytest.raises(LLMConfigError, match="QGEN_EMBED_MODEL=gemini-embedding-001"):
        factory.get_embedder(_settings(embed_provider="openai_compatible", embed_base_url="http://e/v1"))


def test_env_example_with_another_provider_needs_the_eval_model_changed(clean_env):
    """The line QGEN_LLM_MODEL_EVAL=gemini-3.5-flash must be changed with the provider."""
    s = Settings(_env_file=REPO / ".env.example", llm_provider="openai_compatible",
                 llm_base_url="http://a/v1", llm_model="qwen")
    assert s.llm_model_lo == s.llm_model_questions == "qwen"
    with pytest.raises(LLMConfigError, match="QGEN_LLM_MODEL_EVAL"):
        factory.check_config(s, embedder=False)


def test_role_models_follow_inheritance(clean_env):
    s = _settings(llm_model="base", llm_model_questions="q-model")
    assert [factory.model_for_role(r, s) for r in ("lo", "prereq", "questions", "eval", "check", "topics", "split")] == [
        "base", "base", "q-model", "gemini-3.5-flash", "base", "base", "base"]
    inherit_eval = _settings(llm_model="base", llm_model_eval="")
    assert factory.model_for_role("eval", inherit_eval) == "base"


def test_factory_and_retry_require_settings():
    with pytest.raises(TypeError):
        factory.get_llm("lo", None)
    with pytest.raises(TypeError):
        factory.get_embedder(None)
    with pytest.raises(TypeError):
        factory.model_for_role("lo", None)
    with pytest.raises(TypeError):
        retry.generate_json(FakeLLM(["[1]"]), "p", role="lo", model="m", settings=None)
    with pytest.raises(TypeError):
        retry.embed_texts(None, ["a"], model="e", settings=None)


# --- guards ---------------------------------------------------------------------------

ENTRY_POINTS = {"config.py", "cli.py"}  # plus service/ and tools/, which are outside qgen/


def _get_settings_uses(path: Path) -> list[int]:
    """Lines where the code (not a docstring or comment) imports or calls get_settings."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "get_settings":
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "get_settings":
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "get_settings" for alias in node.names):
            lines.append(node.lineno)
    return lines


def test_get_settings_only_in_config_and_entry_points():
    offenders = {
        path.relative_to(QGEN).as_posix(): _get_settings_uses(path)
        for path in QGEN.rglob("*.py")
        if "__pycache__" not in path.parts and path.relative_to(QGEN).as_posix() not in ENTRY_POINTS
    }
    assert {name: lines for name, lines in offenders.items() if lines} == {}


def test_guard_finds_a_call(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text('"""get_settings() in a docstring is fine."""\nfrom qgen.config import get_settings\nx = get_settings()\n',
                      encoding="utf-8")
    assert _get_settings_uses(sample) == [2, 3]


def test_vendor_sdks_are_imported_in_their_adapters():
    import re

    adapters = QGEN / "llm" / "adapters"
    assert re.search(r"^import openai\b", (adapters / "openai_compatible.py").read_text(encoding="utf-8"), re.M)
    assert re.search(r"^import anthropic\b", (adapters / "anthropic.py").read_text(encoding="utf-8"), re.M)


def test_vendor_sdks_are_loaded_only_by_the_factory_on_demand():
    """qgen.llm.factory imports an adapter only for the chosen provider (lazy import)."""
    source = (QGEN / "llm" / "factory.py").read_text(encoding="utf-8")
    top_level = [node for node in ast.parse(source).body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("adapters" in (getattr(node, "module", "") or "") for node in top_level)
