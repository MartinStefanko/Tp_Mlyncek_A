"""Recording wrapper around the LLM client used by the pipeline in qgen/.

Exposes the subset of the `genai.Client` interface that the pipeline uses
(`client.models.generate_content`, `client.models.embed_content`), delegates
every call to the real client and appends one JSON line per call to a JSONL file.

Each record contains:
    seq, method, model, contents (input), config, duration_s,
    and either `output` ({"text": ...} or {"embeddings": [[...], ...]})
    or `error` ({"type": ..., "message": ...}) when the call raised.

Failed calls are recorded too (and re-raised), so a replay client can
reproduce retry behaviour in `generate_with_retry`.

This module does not import any LLM vendor library; the real client is injected.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of call arguments to JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    return repr(value)


def _extract_text(response: Any) -> Any:
    try:
        return response.text
    except Exception as exc:  # .text may raise on blocked / empty candidates
        return {"text_error": f"{type(exc).__name__}: {exc}"}


def _extract_vectors(response: Any) -> list:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        single = getattr(response, "embedding", None)
        embeddings = [single] if single is not None else []
    vectors = []
    for emb in embeddings:
        values = getattr(emb, "values", None)
        if values is None and isinstance(emb, dict):
            values = emb.get("values")
        vectors.append(list(values) if values is not None else None)
    return vectors


class CallRecorder:
    """Thread-safe JSONL writer shared by the wrapper objects."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self.counts: dict[str, int] = {}
        self.errors = 0

    def write(self, record: dict) -> None:
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, **record}
            method = record.get("method", "?")
            self.counts[method] = self.counts.get(method, 0) + 1
            if "error" in record:
                self.errors += 1
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class RecordingModels:
    def __init__(self, real_models: Any, recorder: CallRecorder):
        self._real = real_models
        self._recorder = recorder

    def _call(self, method: str, extract, model: str, contents: Any, config: Any, kwargs: dict):
        record = {
            "method": method,
            "model": model,
            "contents": _to_jsonable(contents),
        }
        if config is not None:
            record["config"] = _to_jsonable(config)
        if kwargs:
            record["kwargs"] = _to_jsonable(kwargs)

        real_method = getattr(self._real, method)
        call_kwargs = dict(kwargs)
        if config is not None:
            call_kwargs["config"] = config

        started = time.perf_counter()
        try:
            response = real_method(model=model, contents=contents, **call_kwargs)
        except Exception as exc:
            record["duration_s"] = round(time.perf_counter() - started, 4)
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._recorder.write(record)
            raise
        record["duration_s"] = round(time.perf_counter() - started, 4)
        record["output"] = extract(response)
        self._recorder.write(record)
        return response

    def generate_content(self, *, model: str, contents: Any, config: Any = None, **kwargs):
        return self._call(
            "generate_content",
            lambda r: {"text": _extract_text(r)},
            model, contents, config, kwargs,
        )

    def embed_content(self, *, model: str, contents: Any, config: Any = None, **kwargs):
        return self._call(
            "embed_content",
            lambda r: {"embeddings": _extract_vectors(r)},
            model, contents, config, kwargs,
        )

    def __getattr__(self, name: str):
        # Anything the pipeline does not use is passed through unrecorded.
        return getattr(self._real, name)


class RecordingClient:
    """Drop-in replacement for `genai.Client` as far as the pipeline uses it."""

    def __init__(self, real_client: Any, jsonl_path: str | Path):
        self._real = real_client
        self.recorder = CallRecorder(jsonl_path)
        self.models = RecordingModels(real_client.models, self.recorder)

    def close(self) -> None:
        self.recorder.close()

    def __getattr__(self, name: str):
        return getattr(self._real, name)
