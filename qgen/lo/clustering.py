import os
from typing import Any, Dict, List, Optional

import numpy as np
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.82
DEBUG_FLOOR = 0.5
DEBUG_TOP = 30
REQUIRE_SOURCE_OVERLAP = False
EMBEDDING_MODEL = "gemini-embedding-001"


def ensure_client(client: Optional[genai.Client] = None) -> genai.Client:
    if client is not None:
        return client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chýba GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


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


def embed_batch(texts: List[str], client: genai.Client, model: str) -> List[List[float]]:
    if not texts:
        return []
    MAX_BATCH = 100
    vectors: List[List[float]] = []
    for start in range(0, len(texts), MAX_BATCH):
        chunk = texts[start : start + MAX_BATCH]
        response = client.models.embed_content(model=model, contents=chunk)
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            single = getattr(response, "embedding", None)
            if single is not None:
                embeddings = [single]
            elif isinstance(response, dict) and "embedding" in response:
                embeddings = [response["embedding"]]
        if embeddings is None:
            raise RuntimeError("API nevrátilo embeddings.")
        if len(embeddings) != len(chunk):
            raise RuntimeError("Počet embeddingov nezodpovedá počtu vstupov.")
        for emb in embeddings:
            values = getattr(emb, "values", None) or (emb.get("values") if isinstance(emb, dict) else None)
            if values is None:
                raise RuntimeError("Embedding bez hodnoty 'values'.")
            vectors.append(list(values))
    return vectors


def cluster_by_core(
    lo_list: List[Dict[str, Any]],
    similarity_threshold: float = SIM_THRESHOLD,
    client: Optional[genai.Client] = None,
    embedding_model: str = EMBEDDING_MODEL,
):
    if len(lo_list) < 2:
        return lo_list

    client = ensure_client(client)

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

        try:
            embeddings = embed_batch(texts, client, embedding_model)
        except Exception as e:
            print(f"[{bloom}] Zlyhalo získanie embeddingov: {e}")
            return lo_list

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
                    if REQUIRE_SOURCE_OVERLAP and not has_source_overlap:
                        continue
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)

        for cluster in clusters:
            representative_idx = cluster[0]
            result.append(items[representative_idx])

        print(f"[{bloom}] Zhlukovanie: {len(items)} -> {len(clusters)} (threshold {similarity_threshold})")

    return result
