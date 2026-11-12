"""Replay: answers LLM / embedding calls from a recording (JSONL) instead of an API.

Reads the file written by ``tools/recording_client.py`` (e.g. baseline_v6/llm_calls.jsonl).
Two users:

    ReplayClient  tests (replay tests on baseline_v6/). Answers are used up; a call without a
                  (remaining) answer raises ``UnrecordedCallError`` (BaseException), which
                  stops the run at once and shows the prompt.
    ReplayLLM     the service in development (QGEN_LLM_PROVIDER=replay and / or
                  QGEN_EMBED_PROVIDER=replay, built by qgen.llm.factory). Never for
                  production: check_config logs a loud warning. A call that is not in the
                  recording raises a ProviderError that is not retried, so the job fails with
                  llm_error like with a real provider. Answers are not used up: a key recorded
                  more than once gives its answers in recorded order and then starts again,
                  so one worker can run the same document many times.

Both implement the two interfaces of the LLM layer (``qgen.llm.base.LLMClient`` and
``EmbeddingClient``), so one object serves as ``llm`` and ``embedder``.

A response is found by its key (method, model, sha256 of the input), not by call order.
The method names are the ones in the recording: ``generate_content`` for ``generate`` and
``embed_content`` for ``embed``.

Optional ``transforms`` (tests only) change the recorded inputs before the key is
computed, to accept a deliberate, named change in a prompt.

Recorded failures: an ``error`` record raises a retryable ``ProviderError`` (as a transient
API error would), except type ``OutputTruncatedError``, which is raised again as it is (the
question pipeline then splits its batch, as in the recorded run); a ``text_error`` record
(blocked / empty answer) gives text "".

ReplayLLM options (QGEN_LLM_REPLAY_*):
    delay_s           sleep before every call, so a container can be killed during a run
    fail_questions    every question generation call fails (ProviderError 400, not retried
                      → LLMError → job failed with llm_error); to test failed + retry.
                      Question generation prompts are recognised by QUESTION_PROMPT_MARKERS
                      (a test keeps them in sync with qgen.questions.pipeline).

This module never calls a real API and imports no LLM vendor library.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from qgen.llm.base import LLMResponse
from qgen.llm.errors import OutputTruncatedError, ProviderError

PREVIEW_CHARS = 1500
GENERATE = "generate_content"
EMBED = "embed_content"

# First lines of the question generation prompt (qgen.questions.prompts.GENERATION_MARKERS,
# sk / en); a test keeps them in sync. Plain strings: this module imports nothing from qgen
# outside qgen.llm.
QUESTION_PROMPT_MARKERS = (
    "Z tematických celkov učebného materiálu a ich textu vytvor otázky a úlohy pre študentov.",
    "From the sections of the study material and their text, create questions and tasks for students.",
)


def input_hash(contents: Any) -> str:
    """Stable hash of a call input (prompt string or list of texts)."""
    payload = json.dumps(contents, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preview(contents: Any, limit: int = PREVIEW_CHARS) -> str:
    text = contents if isinstance(contents, str) else json.dumps(contents, ensure_ascii=False)
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars omitted] ...\n{text[-half:]}"


def is_question_prompt(prompt: str) -> bool:
    return any(marker in prompt for marker in QUESTION_PROMPT_MARKERS)


def read_recording(jsonl_path: str | Path) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class RecordingTransform:
    """A named, documented change applied to recorded inputs before matching.

    ``apply`` gets (method, contents) and returns (new_contents, number_of_changes).
    """

    name: str
    description: str
    apply: Callable[[str, Any], tuple[Any, int]]
    hits: int = field(default=0, compare=False)


class UnrecordedCallError(BaseException):
    """The pipeline made a call that has no (remaining) answer in the recording (tests).

    Derives from BaseException on purpose: the retry layer catches ``Exception``. A missing
    recording must stop a test run at once and show the prompt, not be retried or turned
    into an LLMError.
    """

    def __init__(self, method: str, model: str, contents: Any, key_hash: str, reason: str):
        self.method = method
        self.model = model
        self.contents = contents
        super().__init__(
            f"{reason}: method={method} model={model} input_sha256={key_hash}\n"
            f"--- input preview ---\n{preview(contents)}\n--- end preview ---"
        )


def _raise_recorded_error(record: dict) -> None:
    if "error" in record:
        err = record["error"]
        if err.get("type") == "OutputTruncatedError":
            raise OutputTruncatedError(f"[replayed] {err.get('message')}")
        raise ProviderError(f"[replayed {err.get('type')}] {err.get('message')}", retryable=True)


def _text(record: dict) -> str:
    text = (record.get("output") or {}).get("text")
    return text if isinstance(text, str) else ""  # {"text_error": ...} or None: blocked / empty


def _embeddings(record: dict) -> list[list[float]]:
    return [list(v) for v in (record.get("output") or {}).get("embeddings") or []]


class ReplayClient:
    """LLMClient + EmbeddingClient over a recording, for tests (answers are used up).

    `jsonl_path` may be a list: the recordings are joined in that order (e.g. baseline_v6
    and a short recording of the split step on top of it, tools/baseline.py --base)."""

    def __init__(
        self,
        jsonl_path: str | Path | Sequence[str | Path],
        transforms: Iterable[RecordingTransform] = (),
    ):
        self.transforms = list(transforms)
        self._queues: dict[tuple, deque] = defaultdict(deque)
        self._known_keys: set[tuple] = set()
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []
        paths = [jsonl_path] if isinstance(jsonl_path, (str, Path)) else list(jsonl_path)
        for order, path in enumerate(paths):
            for record in read_recording(path):
                record["_recording"] = order
                contents = self._transform(record["method"], record["contents"])
                key = (record["method"], record["model"], input_hash(contents))
                self._queues[key].append(record)
                self._known_keys.add(key)

    def _transform(self, method: str, contents: Any) -> Any:
        for transform in self.transforms:
            contents, changes = transform.apply(method, contents)
            transform.hits += changes
        return contents

    def _next_record(self, method: str, model: str, contents: Any) -> dict:
        key = (method, model, input_hash(contents))
        with self._lock:
            queue = self._queues.get(key)
            if not queue:
                reason = (
                    "Call was recorded, but all recorded answers were already used"
                    if key in self._known_keys
                    else "Call is not in the recording"
                )
                raise UnrecordedCallError(method, model, contents, key[2], reason)
            record = queue.popleft()
            self.calls.append((method, model))
        return record

    # --- LLMClient ---------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        record = self._next_record(GENERATE, model, prompt)
        _raise_recorded_error(record)
        return LLMResponse(text=_text(record), model=model, usage=dict(record.get("usage") or {}))

    # --- EmbeddingClient ---------------------------------------------------------------
    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        record = self._next_record(EMBED, model, list(texts))
        _raise_recorded_error(record)
        return _embeddings(record)

    def unused_records(self, recording: Optional[int] = None) -> list[dict]:
        """Recorded calls that the run never asked for, in recorded order; `recording`:
        only those of that recording (index in the list given to __init__)."""
        rest = [
            record
            for queue in self._queues.values()
            for record in queue
            if recording is None or record.get("_recording") == recording
        ]
        return sorted(rest, key=lambda r: (r.get("_recording", 0), r.get("seq", 0)))


class ReplayLLM:
    """LLMClient + EmbeddingClient over a recording, for the service in development.

    Answers are not used up (see the module docstring). Errors carry only the method, the
    model and the input hash, never the prompt (logs must not contain document text).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        delay_s: float = 0.0,
        fail_questions: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.path = Path(jsonl_path)
        self.delay_s = delay_s
        self.fail_questions = fail_questions
        self._sleep = sleep
        self._answers: dict[tuple, list[dict]] = defaultdict(list)
        self._next: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
        for record in read_recording(self.path):
            key = (record["method"], record["model"], input_hash(record["contents"]))
            self._answers[key].append(record)

    def _record(self, method: str, model: str, contents: Any) -> dict:
        key = (method, model, input_hash(contents))
        with self._lock:
            answers = self._answers.get(key)
            if not answers:
                raise ProviderError(
                    f"replay: call is not in the recording (method={method}, model={model}, "
                    f"input_sha256={key[2][:16]})",
                    retryable=False,
                    status=404,
                )
            index = self._next[key]
            self._next[key] = (index + 1) % len(answers)
            return answers[index]

    def _wait(self) -> None:
        if self.delay_s > 0:
            self._sleep(self.delay_s)

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        self._wait()
        if self.fail_questions and is_question_prompt(prompt):
            raise ProviderError(
                "replay: question generation fails on purpose (QGEN_LLM_REPLAY_FAIL)", retryable=False, status=400
            )
        record = self._record(GENERATE, model, prompt)
        _raise_recorded_error(record)
        return LLMResponse(text=_text(record), model=model, usage=dict(record.get("usage") or {}))

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self._wait()
        record = self._record(EMBED, model, list(texts))
        _raise_recorded_error(record)
        return _embeddings(record)
