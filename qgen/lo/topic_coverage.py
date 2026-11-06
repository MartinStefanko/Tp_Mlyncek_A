from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from qgen.context_builder import format_segment_label
from qgen.config import Settings, setting_or_default
from qgen.llm.retry import embed_texts, generate_json
from qgen.lo.clustering import normalize_list_field


def analyze_topic_coverage(
    segmenty,
    los,
    llm=None,
    embedder=None,
    topic_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    verbose: bool = True,
    source_max_chars: Optional[int] = None,
    *,
    settings: Settings,
):
    embedding_model = setting_or_default(embedding_model, "embed_model", settings)
    similarity_threshold = setting_or_default(similarity_threshold, "topic_match_threshold", settings)
    report = {
        "topics": [],
        "stats": {
            "topics_total": 0,
            "topics_covered": 0,
            "coverage_percent": 0.0,
            "similarity_threshold": similarity_threshold,
        },
    }

    topics = extract_document_topics(
        segmenty, llm=llm, model=topic_model, verbose=verbose, source_max_chars=source_max_chars,
        settings=settings,
    )
    report["topics"] = topics
    report["stats"]["topics_total"] = len(topics)

    if not topics or not los:
        return report

    topic_texts = [topic["tema"] for topic in topics]
    lo_texts = [lo_to_text(lo) for lo in los]

    # Step 6: no `except Exception` returning the report without coverage; LLMError propagates.
    topic_embeddings = embed_texts(embedder, topic_texts, model=embedding_model, settings=settings)
    lo_embeddings = embed_texts(embedder, lo_texts, model=embedding_model, settings=settings)

    sim_matrix = cosine_similarity(np.array(topic_embeddings), np.array(lo_embeddings))

    covered_count = 0
    for idx, topic in enumerate(topics):
        best_lo_idx = int(np.argmax(sim_matrix[idx]))
        best_score = float(sim_matrix[idx][best_lo_idx])
        best_lo = los[best_lo_idx]
        is_covered = best_score >= similarity_threshold
        if is_covered:
            covered_count += 1

        topic["best_lo_id"] = best_lo.get("id")
        topic["best_lo_name"] = best_lo.get("vzdelávací_objekt", "")
        topic["similarity"] = round(best_score, 4)
        topic["covered"] = is_covered

    report["stats"]["topics_covered"] = covered_count
    report["stats"]["coverage_percent"] = round((covered_count / len(topics)) * 100, 2)
    return report


def extract_document_topics(
    segmenty,
    llm=None,
    model: Optional[str] = None,
    verbose: bool = True,
    source_max_chars: Optional[int] = None,
    *,
    settings: Settings,
):
    model = setting_or_default(model, "llm_model", settings)
    source_text = build_topic_source_text(segmenty, max_chars=source_max_chars)
    if not source_text.strip():
        return []

    prompt = f"""
Si učiteľ. Z nasledujúceho učebného materiálu identifikuj hlavné témy dokumentu.

Vráť LEN validný JSON objekt {{"items": [...]}}, kde "items" je pole objektov v tvare:
{{"items": [
  {{"tema": "Názov témy"}}
]}}

Pravidlá:
- vráť najdôležitejšie témy
- témy majú byť stručné a bez duplicít
- témy majú reprezentovať hlavný obsah dokumentu, nie detaily
- ignoruj obsah, úvod, administratívne pasáže a technické metadáta dokumentu

Materiál:
\"\"\"{source_text}\"\"\"
"""

    # Step 6: no `except Exception: return []` (document without topics); LLMError propagates.
    parsed = generate_json(
        llm, prompt, role="topics", model=model, verbose=verbose, settings=settings, items_key="items"
    )

    seen = set()
    topics = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        tema = str(row.get("tema", "")).strip()
        if not tema:
            continue
        key = tema.lower()
        if key in seen:
            continue
        seen.add(key)
        topics.append({"tema": tema})
    return topics


def build_topic_source_text(segmenty, max_chars: Optional[int] = None):
    max_chars = setting_or_default(max_chars, "topic_source_max_chars")
    parts: List[str] = []
    total_len = 0
    for seg in segmenty:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        page = seg.get("page")
        block = f"[{format_segment_label(seg)}]\n{text}" if page is not None else text
        if total_len + len(block) > max_chars:
            remaining = max_chars - total_len
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total_len += len(block)
    return "\n\n".join(parts)


def lo_to_text(lo: Dict[str, Any]):
    parts = [
        str(lo.get("vzdelávací_objekt", "")).strip(),
        str(lo.get("bloom_level", "")).strip(),
        normalize_list_field(lo.get("odporúčané_aktivity")),
        normalize_list_field(lo.get("odporúčané_zadania")),
    ]
    return " ".join(part for part in parts if part)
