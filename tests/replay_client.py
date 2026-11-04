"""Replay client: answers LLM / embedding calls from a recording (JSONL) instead of an API.

Reads the file written by ``tools/recording_client.py``. ``ReplayClient`` implements both
interfaces of the LLM layer (``qgen.llm.base.LLMClient`` and ``EmbeddingClient``), so a test
passes the same object as ``llm`` and ``embedder``.

A response is found by its key (method, model, sha256 of the input), not by call order.
The method names are the ones in the recording: ``generate_content`` for ``generate`` and
``embed_content`` for ``embed``. So a call that was removed from the pipeline (e.g. Python
document classification) does not shift the other answers. When the same input was
recorded more than once, the answers are returned in the recorded order. A call without a
recorded answer raises ``UnrecordedCallError`` with a preview of the prompt.

Optional ``transforms`` change the recorded inputs before the key is computed. They let a
test accept a deliberate, named change in a prompt (the step 4 transforms of the former
gemini-2.5 baseline/ were removed with it; the replay tests on baseline_v3/ use none).

Recorded failures: an ``error`` record raises a retryable ``ProviderError`` (as a transient
API error would); a ``text_error`` record (blocked / empty answer) gives text "".

This module never calls a real API and imports no LLM vendor library.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from qgen.llm.base import LLMResponse
from qgen.llm.errors import ProviderError

PREVIEW_CHARS = 1500
GENERATE = "generate_content"
EMBED = "embed_content"


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
    """The pipeline made a call that has no (remaining) answer in the recording.

    Derives from BaseException on purpose: the retry layer catches ``Exception``. A missing
    recording must stop the run at once and show the prompt, not be retried or turned into
    an LLMError.
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
        raise ProviderError(f"[replayed {err.get('type')}] {err.get('message')}", retryable=True)


class ReplayClient:
    """LLMClient + EmbeddingClient over a recording."""

    def __init__(self, jsonl_path: str | Path, transforms: Iterable[RecordingTransform] = ()):
        self.transforms = list(transforms)
        self._queues: dict[tuple, deque] = defaultdict(deque)
        self._known_keys: set[tuple] = set()
        self.calls: list[tuple[str, str]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
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
        text = (record.get("output") or {}).get("text")
        if not isinstance(text, str):  # {"text_error": ...} or None: blocked / empty answer
            text = ""
        return LLMResponse(text=text, model=model, usage=dict(record.get("usage") or {}))

    # --- EmbeddingClient ---------------------------------------------------------------
    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        record = self._next_record(EMBED, model, list(texts))
        _raise_recorded_error(record)
        return [list(v) for v in (record.get("output") or {}).get("embeddings") or []]

    def unused_records(self) -> list[dict]:
        """Recorded calls that the run never asked for, in recorded order."""
        rest = [record for queue in self._queues.values() for record in queue]
        return sorted(rest, key=lambda r: r.get("seq", 0))
