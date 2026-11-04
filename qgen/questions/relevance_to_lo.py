from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from lo_clustering import EMBEDDING_MODEL, embed_batch, ensure_client, normalize_list_field


def analyze_item_relevance_to_lo(
    items,
    los,
    client=None,
    embedding_model: str = EMBEDDING_MODEL,
    verbose: bool = True,
):
    report = {
        "stats": {
            "items_total": len(items),
            "items_compared": 0,
            "average_similarity": 0.0,
        },
        "items": [],
    }

    if not items or not los:
        return report

    lo_map = {lo.get("id"): lo for lo in los if isinstance(lo.get("id"), int)}
    comparable_items: List[Dict[str, Any]] = []

    for item in items:
        lo_id = item.get("lo_id")
        lo = lo_map.get(lo_id)

        row = {
            "item_id": item.get("id"),
            "lo_id": lo_id,
            "lo_name": lo.get("vzdelávací_objekt", "") if lo else "",
            "similarity": None,
            "has_lo": lo is not None,
        }
        report["items"].append(row)

        if lo is None:
            continue

        item_text = item_to_text(item)
        lo_text = lo_to_text(lo)
        if not item_text.strip() or not lo_text.strip():
            continue

        row["item_text"] = item_text
        row["lo_text"] = lo_text
        comparable_items.append(row)

    if not comparable_items:
        return report

    client = ensure_client(client)

    try:
        item_embeddings = embed_batch([row["item_text"] for row in comparable_items], client, embedding_model)
        lo_embeddings = embed_batch([row["lo_text"] for row in comparable_items], client, embedding_model)
    except Exception as e:
        if verbose:
            print(f"Analyza relevance polozky k LO zlyhala pri embeddingoch: {e}")
        for row in report["items"]:
            row.pop("item_text", None)
            row.pop("lo_text", None)
        return report

    similarities = []
    for idx, row in enumerate(comparable_items):
        sim = float(cosine_similarity(np.array([item_embeddings[idx]]), np.array([lo_embeddings[idx]]))[0][0])
        row["similarity"] = round(sim, 4)
        similarities.append(sim)

    report["stats"]["items_compared"] = len(similarities)
    report["stats"]["average_similarity"] = round(sum(similarities) / len(similarities), 4)

    for row in report["items"]:
        row.pop("item_text", None)
        row.pop("lo_text", None)

    return report


def item_to_text(item: Dict[str, Any]):
    parts = [
        str(item.get("typ", "")).strip(),
        str(item.get("otazka", "")).strip(),
        normalize_list_field(item.get("odpoved")),
        normalize_list_field(item.get("napoveda")),
    ]
    return " ".join(part for part in parts if part)


def lo_to_text(lo: Dict[str, Any]):
    parts = [
        str(lo.get("vzdelávací_objekt", "")).strip(),
        str(lo.get("bloom_level", "")).strip(),
        normalize_list_field(lo.get("odporúčané_aktivity")),
        normalize_list_field(lo.get("odporúčané_zadania")),
    ]
    return " ".join(part for part in parts if part)
