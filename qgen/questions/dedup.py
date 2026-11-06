"""Duplicate questions within one section (tematický celok), by embeddings.

Text of a question: description + the texts of its correct choices (no_answer: only the
description). A new question whose cosine similarity with a kept question of the same
section is >= QGEN_QUESTION_DEDUP_THRESHOLD is a duplicate; questions are taken in the order
they were created, so the first wording stays.

Calibration (step 13, baseline_v3, recorded embeddings of the questions): within sections
the median similarity is 0.81, the 95th percentile 0.91, the maximum 0.938 (two different
questions on the same concept). The default 0.95 removes none of them. tools/dedup_report.py
shows the most similar pairs of a recording to tune the threshold on baseline_v4.
"""

from __future__ import annotations

import math
from typing import Optional

from qgen.config import Settings
from qgen.llm.base import EmbeddingClient
from qgen.llm.retry import embed_texts
from qgen.questions.types import correct_texts


def dedup_text(item: dict) -> str:
    parts = [str(item.get("description", "")).strip()]
    if item.get("type") != "no_answer":
        parts.extend(t.strip() for t in correct_texts(item) if t.strip())
    return " ".join(p for p in parts if p)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SectionDeduplicator:
    """Keeps the vectors of the accepted questions of every section."""

    def __init__(self, threshold: float, embedder: Optional[EmbeddingClient], *, settings: Settings,
                 verbose: bool = False):
        self.threshold = threshold
        self.embedder = embedder
        self.settings = settings
        self.verbose = verbose
        self.kept: dict = {}  # lo_id -> list of vectors

    def filter(self, items: list[dict]) -> tuple[list[dict], list[dict]]:
        """(new, duplicates). The new questions are added to the kept ones."""
        if not items:
            return [], []
        vectors = embed_texts(
            self.embedder, [dedup_text(item) for item in items],
            model=self.settings.embed_model, settings=self.settings, verbose=self.verbose,
        )
        new, duplicates = [], []
        for item, vector in zip(items, vectors):
            kept = self.kept.setdefault(item.get("lo_id"), [])
            if any(cosine(vector, other) >= self.threshold for other in kept):
                duplicates.append(item)
                continue
            kept.append(vector)
            new.append(item)
        return new, duplicates
