"""Replay client: answers LLM / embedding calls from a recording (JSONL) instead of an API.

Reads the file written by ``tools/recording_client.py`` and exposes the same subset of the
client interface that the pipeline uses (``client.models.generate_content`` and
``client.models.embed_content``).

A response is found by its key (method, model, sha256 of the input), not by call order.
So a call that was removed from the pipeline (e.g. Python document classification) does
not shift the other answers. When the same input was recorded more than once, the answers
are returned in the recorded order. A call without a recorded answer raises
``UnrecordedCallError`` with a preview of the prompt.

Optional ``transforms`` change the recorded inputs before the key is computed. They let a
test accept a deliberate, named change in a prompt (see ``tests/recording_transforms.py``).

This module never calls a real API and imports no LLM vendor library.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

PREVIEW_CHARS = 1500


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

    Derives from BaseException on purpose: generate_with_retry and the pipeline catch
    ``Exception`` (retry with sleep, then fall back to an empty result). A missing recording
    must stop the run at once and show the prompt, not be retried or hidden.
    """

    def __init__(self, method: str, model: str, contents: Any, key_hash: str, reason: str):
        self.method = method
        self.model = model
        self.contents = contents
        super().__init__(
            f"{reason}: method={method} model={model} input_sha256={key_hash}\n"
            f"--- input preview ---\n{preview(contents)}\n--- end preview ---"
        )


class _FailingText:
    """Response whose ``.text`` raises, like a blocked / empty candidate in the real SDK."""

    def __init__(self, message: str):
        self._message = message

    @property
    def text(self):
        raise ValueError(self._message)


def _build_response(method: str, record: dict) -> Any:
    if "error" in record:
        err = record["error"]
        raise RuntimeError(f"[replayed {err.get('type')}] {err.get('message')}")
    output = record.get("output") or {}
    if method == "generate_content":
        text = output.get("text")
        if isinstance(text, dict) and "text_error" in text:
            return _FailingText(text["text_error"])
        return SimpleNamespace(text=text)
    if method == "embed_content":
        vectors = output.get("embeddings") or []
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])
    raise ValueError(f"Unsupported recorded method: {method}")


class ReplayModels:
    def __init__(self, client: "ReplayClient"):
        self._client = client

    def generate_content(self, *, model: str, contents: Any, config: Any = None, **kwargs):
        return self._client._answer("generate_content", model, contents)

    def embed_content(self, *, model: str, contents: Any, config: Any = None, **kwargs):
        return self._client._answer("embed_content", model, contents)


class ReplayClient:
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
        self.models = ReplayModels(self)

    def _transform(self, method: str, contents: Any) -> Any:
        for transform in self.transforms:
            contents, changes = transform.apply(method, contents)
            transform.hits += changes
        return contents

    def _answer(self, method: str, model: str, contents: Any) -> Any:
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
        return _build_response(method, record)

    def unused_records(self) -> list[dict]:
        """Recorded calls that the run never asked for, in recorded order."""
        rest = [record for queue in self._queues.values() for record in queue]
        return sorted(rest, key=lambda r: r.get("seq", 0))
