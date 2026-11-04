from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from qgen.config import setting_or_default
from qgen.llm.retry import embed_texts

# Only for the debug print of the most similar pairs; no effect on the result.
DEBUG_FLOOR = 0.5
DEBUG_TOP = 30


def normalize_list_field(val) -> str:
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return " ".join(map(str, val))
    return str(val)


def normalize_sources(val) -> set:
    if val is None:
        return set()
    if not isinstance(val, (list, tuple)):
        val = [val]
    cleaned = []
    for v in val:
        s = str(v).strip().lower()
        if s:
            cleaned.append(s)
    return set(cleaned)


def cluster_by_core(
    lo_list: List[Dict[str, Any]],
    similarity_threshold: Optional[float] = None,
    embedder: Optional[Any] = None,
    embedding_model: Optional[str] = None,
    require_source_overlap: Optional[bool] = None,
):
    if len(lo_list) < 2:
        return lo_list

    similarity_threshold = setting_or_default(similarity_threshold, "cluster_sim_threshold")
    embedding_model = setting_or_default(embedding_model, "embed_model")
    require_source_overlap = setting_or_default(require_source_overlap, "cluster_require_source_overlap")

    bloom_groups: Dict[str, List[Dict[str, Any]]] = {}
    for obj in lo_list:
        bloom = str(obj.get("bloom_level", "")).strip()
        bloom_groups.setdefault(bloom, []).append(obj)

    result = []
    for bloom, items in bloom_groups.items():
        if len(items) == 1:
            result.extend(items)
            continue

        texts = []
        for o in items:
            text_parts = [
                str(o.get("vzdelávací_objekt", "")),
                normalize_list_field(o.get("odporúčané_zadania")),
                #normalize_list_field(o.get("odporúčané_aktivity")),
            ]
            texts.append(" ".join(text_parts))

        # Step 6: no `except Exception: return lo_list` (unclustered LO); LLMError propagates.
        embeddings = embed_texts(embedder, texts, model=embedding_model)

        sim_matrix = cosine_similarity(np.array(embeddings))

        pairs = []
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i, j]
                if sim >= DEBUG_FLOOR:
                    pairs.append((sim, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        if pairs:
            print(f"[{bloom}] Top podobnosti (>= {DEBUG_FLOOR}):")
            for sim, i, j in pairs[:DEBUG_TOP]:
                name_i = items[i].get("vzdelávací_objekt", "")
                name_j = items[j].get("vzdelávací_objekt", "")
                print(f"  {sim:.3f} - {name_i}  <->  {name_j}")
        else:
            print(f"[{bloom}] Nenašli sa páry s podobnosťou >= {DEBUG_FLOOR}")

        visited = set()
        clusters = []
        for i in range(n):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, n):
                if j in visited:
                    continue
                sim = sim_matrix[i, j]
                sources_i = normalize_sources(items[i].get("citovane_zdroje"))
                sources_j = normalize_sources(items[j].get("citovane_zdroje"))
                has_source_overlap = bool(sources_i & sources_j)

                if sim >= similarity_threshold:
                    if require_source_overlap and not has_source_overlap:
                        continue
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)

        for cluster in clusters:
            representative_idx = cluster[0]
            result.append(items[representative_idx])

        print(f"[{bloom}] Zhlukovanie: {len(items)} -> {len(clusters)} (threshold {similarity_threshold})")

    return result
