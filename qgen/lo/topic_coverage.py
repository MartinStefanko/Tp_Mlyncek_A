from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from json_load import safe_load_json
from context_builder import format_segment_label
from llm_client import generate_with_retry
from lo_clustering import EMBEDDING_MODEL, embed_batch, ensure_client, normalize_list_field

TOPIC_EXTRACTION_MODEL = "gemini-2.5-flash-lite"
TOPIC_MATCH_THRESHOLD = 0.72
MAX_SOURCE_CHARS = 20000


def analyze_topic_coverage(
    segmenty,
    los,
    client=None,
    topic_model: str = TOPIC_EXTRACTION_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    similarity_threshold: float = TOPIC_MATCH_THRESHOLD,
    verbose: bool = True,
):
    report = {
        "topics": [],
        "stats": {
            "topics_total": 0,
            "topics_covered": 0,
            "coverage_percent": 0.0,
            "similarity_threshold": similarity_threshold,
        },
    }

    topics = extract_document_topics(segmenty, client=client, model=topic_model, verbose=verbose)
    report["topics"] = topics
    report["stats"]["topics_total"] = len(topics)

    if not topics or not los:
        return report

    client = ensure_client(client)

    topic_texts = [topic["tema"] for topic in topics]
    lo_texts = [lo_to_text(lo) for lo in los]

    try:
        topic_embeddings = embed_batch(topic_texts, client, embedding_model)
        lo_embeddings = embed_batch(lo_texts, client, embedding_model)
    except Exception as e:
        if verbose:
            print(f"Analyza pokrytia tem zlyhala pri embeddingoch: {e}")
        return report

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


def extract_document_topics(segmenty, client=None, model: str = TOPIC_EXTRACTION_MODEL, verbose: bool = True):
    source_text = build_topic_source_text(segmenty)
    if not source_text.strip():
        return []

    prompt = f"""
Si učiteľ. Z nasledujúceho učebného materiálu identifikuj hlavné témy dokumentu.

Vráť LEN validný JSON ako pole objektov v tvare:
[
  {{"tema": "Názov témy"}}
]

Pravidlá:
- vráť najdôležitejšie témy
- témy majú byť stručné a bez duplicít
- témy majú reprezentovať hlavný obsah dokumentu, nie detaily
- ignoruj obsah, úvod, administratívne pasáže a technické metadáta dokumentu

Materiál:
\"\"\"{source_text}\"\"\"
"""

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Extrakcia hlavnych tem zlyhala: {e}")
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

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


def build_topic_source_text(segmenty):
    parts: List[str] = []
    total_len = 0
    for seg in segmenty:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        page = seg.get("page")
        block = f"[{format_segment_label(seg)}]\n{text}" if page is not None else text
        if total_len + len(block) > MAX_SOURCE_CHARS:
            remaining = MAX_SOURCE_CHARS - total_len
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
