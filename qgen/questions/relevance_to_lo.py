from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from qgen.config import Settings, setting_or_default
from qgen.llm.retry import embed_texts
from qgen.lo.clustering import normalize_list_field
from qgen.questions.types import correct_texts


def analyze_item_relevance_to_lo(
    items,
    los,
    embedder=None,
    embedding_model: Optional[str] = None,
    verbose: bool = True,
    *,
    settings: Settings,
):
    embedding_model = setting_or_default(embedding_model, "embed_model", settings)
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

    # Step 6: no `except Exception` returning an empty report; LLMError propagates.
    item_embeddings = embed_texts(embedder, [row["item_text"] for row in comparable_items], model=embedding_model, settings=settings)
    lo_embeddings = embed_texts(embedder, [row["lo_text"] for row in comparable_items], model=embedding_model, settings=settings)

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
    """Type, question, texts of the correct choices and the hints."""
    parts = [
        str(item.get("type", "")).strip(),
        str(item.get("description", "")).strip(),
        normalize_list_field(correct_texts(item)),
        normalize_list_field([h.get("url", "") for h in item.get("hints") or [] if isinstance(h, dict)]),
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
