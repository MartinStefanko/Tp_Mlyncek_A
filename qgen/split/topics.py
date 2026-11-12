"""Topics of the sections and the topic runs of the document (kap. 8, step c; mode topics).

The caller gives the document topics (names) and the similarity of every section to every
topic (cosine of the embeddings the pipeline already has, step 16). Here only:

    section_topics   topic of every section = the most similar one (spec c, used in the report)
    topic_segments   the order of step a cut into contiguous runs of one topic

The most similar topic changes very often along the document (eko.pdf: 23 runs of mostly one
section for 8 topics), so the runs come from a Viterbi path: maximal Σ similarity(section,
topic of its run) − TOPIC_SWITCH_PENALTY × number of topic changes. The order of the sections
stays; a topic may come back later.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from qgen.split.sections import SplitSection

# Cost of one change of topic, in units of cosine similarity (eko.pdf: 0.1 gives 8 runs).
TOPIC_SWITCH_PENALTY = 0.1

TopicSimilarity = Mapping[int, Sequence[float]]  # section id → similarity to each topic

__all__ = ["TOPIC_SWITCH_PENALTY", "TopicSimilarity", "section_topics", "topic_segments"]


def section_topics(
    sections: Sequence[SplitSection], topic_count: int, similarity: Optional[TopicSimilarity]
) -> dict[int, int]:
    """Section id → index of its most similar topic (the first one on a tie). Sections
    without a similarity row get no topic."""
    found: dict[int, int] = {}
    if not topic_count or similarity is None:
        return found
    for section in sections:
        row = _row(similarity, section.id, topic_count)
        if row is not None:
            found[section.id] = max(range(topic_count), key=lambda k: (row[k], -k))
    return found


def topic_segments(
    ordered: Sequence[SplitSection],
    topic_count: int,
    similarity: Optional[TopicSimilarity],
    penalty: float = TOPIC_SWITCH_PENALTY,
) -> list[tuple[int, list[SplitSection]]]:
    """Contiguous runs (topic index, sections) of the sections in the given order.

    A section without a similarity row counts 0 for every topic, so it stays in the run
    around it. On a tie the path keeps its topic, then takes the lower topic index, so the
    result is deterministic. No topics → [] (the caller decides).
    """
    if not ordered or not topic_count or similarity is None:
        return []
    rows = [_row(similarity, s.id, topic_count) or [0.0] * topic_count for s in ordered]
    scores = list(rows[0])
    back: list[list[int]] = []
    for row in rows[1:]:
        best_prev = max(range(topic_count), key=lambda k: (scores[k], -k))
        new_scores = []
        pointers = []
        for k in range(topic_count):
            stay = scores[k]
            switch = scores[best_prev] - penalty
            if stay >= switch:
                new_scores.append(stay + row[k])
                pointers.append(k)
            else:
                new_scores.append(switch + row[k])
                pointers.append(best_prev)
        scores = new_scores
        back.append(pointers)
    k = max(range(topic_count), key=lambda j: (scores[j], -j))
    path = [k]
    for pointers in reversed(back):
        k = pointers[k]
        path.append(k)
    path.reverse()

    segments: list[tuple[int, list[SplitSection]]] = []
    for section, topic in zip(ordered, path):
        if segments and segments[-1][0] == topic:
            segments[-1][1].append(section)
        else:
            segments.append((topic, [section]))
    return segments


def _row(similarity: TopicSimilarity, section_id: int, topic_count: int) -> Optional[list[float]]:
    row = similarity.get(section_id)
    if row is None:
        return None
    if len(row) != topic_count:
        raise ValueError(f"similarity of section {section_id} has {len(row)} values, expected {topic_count}")
    return [float(x) for x in row]
