"""Recording wrapper around the LLM interface used by the pipeline in qgen/.

`RecordingClient` implements `qgen.llm.base.LLMClient` and `EmbeddingClient`, delegates
every call to the real clients (from qgen.llm.factory) and appends one JSON line per call
to a JSONL file. Since step 6 it wraps the interface, not `genai.Client`; the file format
is unchanged, so old recordings stay valid for qgen/llm/adapters/replay.py.

Each record contains:
    seq, method ("generate_content" | "embed_content"), model, contents (input),
    duration_s, and either `output` ({"text": ...} or {"embeddings": [[...], ...]}, plus
    `usage` when the provider returned token counts) or `error` ({"type", "message"})
    when the call raised.

Failed calls are recorded too (and re-raised), so a replay client can reproduce the
retries of qgen.llm.retry.

This module does not import any LLM vendor library; the real clients are injected.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

GENERATE = "generate_content"
EMBED = "embed_content"


class CallRecorder:
    """Thread-safe JSONL writer."""

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


class RecordingClient:
    """LLMClient + EmbeddingClient that records every call of the wrapped clients."""

    def __init__(self, llm: Any, embedder: Any, jsonl_path: str | Path):
        self._llm = llm
        self._embedder = embedder
        self.recorder = CallRecorder(jsonl_path)

    def _record(self, method: str, model: str, contents: Any, call, extract, config: Optional[dict] = None):
        record = {"method": method, "model": model, "contents": contents}
        if config:
            record["config"] = config  # informative only; the replay key is method, model, contents
        started = time.perf_counter()
        try:
            result = call()
        except Exception as exc:
            record["duration_s"] = round(time.perf_counter() - started, 4)
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.recorder.write(record)
            raise
        record["duration_s"] = round(time.perf_counter() - started, 4)
        record.update(extract(result))
        self.recorder.write(record)
        return result

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        def extract(response):
            data = {"output": {"text": response.text}}
            if response.usage:
                data["usage"] = dict(response.usage)
            return data

        return self._record(
            GENERATE, model, prompt,
            lambda: self._llm.generate(
                prompt, model=model, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens
            ),
            extract,
            config={"json_mode": json_mode, "temperature": temperature, "max_tokens": max_tokens},
        )

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        return self._record(
            EMBED, model, list(texts),
            lambda: self._embedder.embed(texts, model=model),
            lambda vectors: {"output": {"embeddings": [list(v) for v in vectors]}},
        )

    def close(self) -> None:
        self.recorder.close()
