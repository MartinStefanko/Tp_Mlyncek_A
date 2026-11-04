import os
import time
from typing import Any, List, Optional

from google import genai
from dotenv import load_dotenv


def get_client(api_key=None):
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=key)


def generate_with_retry(prompt, client=None, model="gemini-2.5-flash-lite", retries=5, delay=3.0, verbose=True):
    if client is None:
        client = get_client()

    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt
            )
        except Exception as e:
            if attempt == retries:
                raise
            if verbose:
                print(f"LLM zlyhalo (pokus {attempt}/{retries}), cakam {delay}s a opakujem. Chyba: {e}")
            time.sleep(delay)


def ensure_client(client: Optional[Any] = None) -> Any:
    if client is not None:
        return client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chýba GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def embed_batch(texts: List[str], client: Any, model: str) -> List[List[float]]:
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
