"""Scripted fakes of the LLM interface for unit tests. No network, no vendor library.

FakeLLM / FakeEmbedder take a list of steps, used one per call in order; when the list is
used up, `default` answers every further call. A step is:
    str                 FakeLLM: the answer text (may be invalid JSON)
    list of vectors     FakeEmbedder: the answer
    BaseException       raised (see the helpers below)
    callable            called with (prompt_or_texts, model); returns one of the above

Default answer of FakeLLM: '{"items": []}' (every pipeline prompt asks for that object).

Helpers for the errors the retry layer must handle:
    server_error(503)          retryable ProviderError
    client_error(404)          not retryable ProviderError (400, 401, 403, 404, ...)
    rate_limited(retry_after)  RateLimitError (HTTP 429)
    timed_out()                LLMTimeoutError (no real waiting)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from typing import Any, Iterable, Optional

from qgen.llm.base import LLMResponse
from qgen.llm.errors import LLMTimeoutError, OutputTruncatedError, ProviderError, RateLimitError

INVALID_JSON = "Sorry, here is no JSON {"
EMPTY_ITEMS = '{"items": []}'


def server_error(status: int = 503) -> ProviderError:
    return ProviderError(f"fake server error {status}", retryable=True, status=status)


def client_error(status: int = 400) -> ProviderError:
    return ProviderError(f"fake client error {status}", retryable=False, status=status)


def rate_limited(retry_after: Optional[float] = None) -> RateLimitError:
    return RateLimitError("fake 429", retry_after=retry_after)


def timed_out() -> LLMTimeoutError:
    return LLMTimeoutError("fake timeout")


def truncated() -> OutputTruncatedError:
    return OutputTruncatedError("fake answer cut off at the token limit")


def items(rows) -> str:
    """JSON answer {"items": rows} as the prompts ask for it."""
    return json.dumps({"items": list(rows)}, ensure_ascii=False)


# --- the question pipeline (step 13) ---------------------------------------------------------


def section_ids(prompt: str) -> list[int]:
    """lo_id of the section blocks of a question generation prompt (not the existing list)."""
    return [int(x) for x in re.findall(r"^- lo_id: (\d+)$", prompt, re.MULTILINE)]


def is_question_generation(prompt: str) -> bool:
    return "vytvor otázky a úlohy pre študentov" in prompt or "create questions and tasks for students" in prompt


def is_extra_round(prompt: str) -> bool:
    return "DOPLŇUJÚCE KOLO" in prompt or "ADDITIONAL ROUND" in prompt


def is_coverage_call(prompt: str) -> bool:
    return "DOPLNENIE POKRYTIA" in prompt or "COVERAGE SUPPLEMENT" in prompt


def question(lo_id: int, n: int = 1, question_type: str = "choice", **fields) -> dict:
    """One valid generated question in the format of the generation prompt."""
    text = f"Otázka {n} k celku {lo_id}?"
    if question_type == "choice":
        choices = [
            {"text": f"Správna {n}/{lo_id}", "is_correct": True},
            {"text": f"Zlá A {n}/{lo_id}", "is_correct": False},
            {"text": f"Zlá B {n}/{lo_id}", "is_correct": False},
            {"text": f"Zlá C {n}/{lo_id}", "is_correct": False},
        ]
    elif question_type == "open":
        choices = [{"text": f"Odpoveď {n}/{lo_id}.", "is_correct": True}]
    else:
        choices = []
    row = {
        "lo_id": lo_id,
        "name": f"Otázka {n}",
        "description": text,
        "type": question_type,
        "practical": False,
        "choices": choices,
        "hints": ["<p>Zamerajte sa na definíciu.</p>", "<p>Porovnajte oba pojmy.</p>"],
        "time_to_handle": 60 if question_type == "choice" else 300,
        "time_to_explain": 30 if question_type == "choice" else 60,
        "bloom_level": "Pochopiť",
        "keywords": ["pojem"],
        "source": [f"D1:{lo_id}"],
    }
    row.update(fields)
    return row


def check_answer(prompt: str, score: int = 5) -> Optional[str]:
    """Answer of the section and question checks with `score`; None for other prompts."""
    if "Item id:" in prompt:  # question quality score
        ids = [int(x) for x in re.findall(r"Item id: (\d+)", prompt)]
        return items({"id": i, "skore": score, "zdovodnenie": "ok"} for i in ids)
    if "ITEM ID:" in prompt:  # question faithfulness / answerability
        ids = [int(x) for x in re.findall(r"ITEM ID: (\d+)", prompt)]
        return items({"item_id": i, "skore": score, "zdovodnenie": "ok", "choices_ok": True, "wrong_choices": []}
                     for i in ids)
    if "LO ID:" in prompt:  # section faithfulness
        ids = [int(x) for x in re.findall(r"LO ID: (\d+)", prompt)]
        return items({"lo_id": i, "skore": score, "zdovodnenie": "ok"} for i in ids)
    return None


def _resolve(step: Any, arg: Any, model: str) -> Any:
    if callable(step) and not isinstance(step, BaseException):
        step = step(arg, model)
    if isinstance(step, BaseException):
        raise step
    return step


class FakeLLM:
    """LLMClient with scripted answers. `calls` lists (model, prompt) of every call."""

    def __init__(self, steps: Iterable[Any] = (), *, default: Any = EMPTY_ITEMS):
        self.steps = deque(steps)
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        self.calls.append((model, prompt))
        step = self.steps.popleft() if self.steps else self.default
        text = _resolve(step, prompt, model)
        return LLMResponse(text=text, model=model, usage={"input_tokens": len(prompt)})


class OrthogonalVectors:
    """Embedding answer: every distinct text gets its own one-hot vector (no clustering,
    no duplicates); `same` maps texts to a shared key, so those texts get equal vectors."""

    def __init__(self, dim: int = 512, same: Optional[dict] = None) -> None:
        self.dim = dim
        self.same = same if same is not None else {}
        self.index: dict[str, int] = {}

    def __call__(self, texts, model):
        vectors = []
        for text in texts:
            key = self.same.get(text, text)
            position = self.index.setdefault(key, len(self.index) % self.dim)
            vectors.append([1.0 if i == position else 0.0 for i in range(self.dim)])
        return vectors


def hash_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic pseudo embedding of a text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:dim]]


class FakeEmbedder:
    """EmbeddingClient with scripted answers; default: hash vectors. `calls` lists the inputs."""

    def __init__(self, steps: Iterable[Any] = (), *, default: Any = None):
        self.steps = deque(steps)
        self.default = default
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.steps:
            step = self.steps.popleft()
        elif self.default is not None:
            step = self.default
        else:
            return [hash_vector(t) for t in texts]
        return _resolve(step, texts, model)
