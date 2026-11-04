from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from context_builder import build_context_for_lo, build_page_map
from lo_clustering import EMBEDDING_MODEL, embed_batch, ensure_client, normalize_list_field


def analyze_lo_relevance_to_segment(
    segmenty,
    los,
    client=None,
    embedding_model: str = EMBEDDING_MODEL,
    verbose: bool = True,
):
    report = {
        "stats": {
            "los_total": len(los),
            "los_compared": 0,
            "average_similarity": 0.0,
        },
        "items": [],
    }

    if not los:
        return report

    page_map = build_page_map(segmenty)
    comparable_items: List[Dict[str, Any]] = []

    for lo in los:
        lo_text = lo_to_text(lo)
        source_text = build_context_for_lo(lo, page_map, max_chars=8000)

        item = {
            "lo_id": lo.get("id"),
            "lo_name": lo.get("vzdelávací_objekt", ""),
            "similarity": None,
            "has_source_text": bool(source_text.strip()),
            "source_pages": lo.get("citovane_zdroje", []),
        }

        report["items"].append(item)

        if not lo_text.strip() or not source_text.strip():
            continue

        item["lo_text"] = lo_text
        item["source_text"] = source_text
        comparable_items.append(item)

    if not comparable_items:
        return report

    client = ensure_client(client)

    try:
        lo_embeddings = embed_batch([item["lo_text"] for item in comparable_items], client, embedding_model)
        source_embeddings = embed_batch([item["source_text"] for item in comparable_items], client, embedding_model)
    except Exception as e:
        if verbose:
            print(f"Analyza LO vs zdrojovy segment zlyhala pri embeddingoch: {e}")
        for item in report["items"]:
            item.pop("lo_text", None)
            item.pop("source_text", None)
        return report

    similarities = []
    for idx, item in enumerate(comparable_items):
        sim = float(cosine_similarity(np.array([lo_embeddings[idx]]), np.array([source_embeddings[idx]]))[0][0])
        item["similarity"] = round(sim, 4)
        similarities.append(sim)

    report["stats"]["los_compared"] = len(similarities)
    report["stats"]["average_similarity"] = round(sum(similarities) / len(similarities), 4)

    for item in report["items"]:
        item.pop("lo_text", None)
        item.pop("source_text", None)

    return report


def lo_to_text(lo: Dict[str, Any]):
    parts = [
        str(lo.get("vzdelávací_objekt", "")).strip(),
        str(lo.get("bloom_level", "")).strip(),
        normalize_list_field(lo.get("odporúčané_aktivity")),
        normalize_list_field(lo.get("odporúčané_zadania")),
    ]
    return " ".join(part for part in parts if part)
