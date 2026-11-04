from json_load import safe_load_json
from llm_client import generate_with_retry


SUPPORTED_DOCUMENT_LANGUAGES = {"sk", "en"}


def detect_document_language(segmenty, client=None, model="gemini-2.5-flash-lite", verbose=True):
    source_text = build_full_document_text(segmenty)
    if not source_text.strip():
        return {"language": "sk", "reason": "Dokument nema text pre klasifikaciu."}

    prompt = f"""
Detect whether the following document is primarily written in Slovak or English.

Return ONLY valid JSON in this format:
{{
  "language": "sk",
  "reason": "short justification"
}}

Rules:
- return "sk" if the document is primarily in Slovak
- return "en" if the document is primarily in English
- if the document is mixed, return the dominant language
- allowed values for "language" are only "sk" or "en"

Document:
\"\"\"{source_text}\"\"\"
"""

    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
    except Exception as e:
        if verbose:
            print(f"Detekcia jazyka dokumentu zlyhala: {e}")
        return heuristic_language_fallback(source_text, "Detekcia zlyhala, pouzity heuristicky fallback.")

    if not isinstance(parsed, dict):
        return heuristic_language_fallback(source_text, "Neplatna odpoved detekcie, pouzity heuristicky fallback.")

    language = str(parsed.get("language", "")).strip().lower()
    if language not in SUPPORTED_DOCUMENT_LANGUAGES:
        return heuristic_language_fallback(source_text, "Nepodporovany jazyk z detekcie, pouzity heuristicky fallback.")

    return {
        "language": language,
        "reason": str(parsed.get("reason", "")).strip(),
    }


def heuristic_language_fallback(text, reason_prefix):
    lowered = str(text or "").lower()
    sk_tokens = [
        " je ", " a ", " že ", " pre ", " ako ", " ktorý ", " ktoré ", " alebo ",
        " vzdel", " strán", " dokument", " úloha", " otázka",
    ]
    en_tokens = [
        " the ", " and ", " is ", " are ", " for ", " with ", " from ", " learning ",
        " document ", " page ", " question ", " task ",
    ]
    sk_score = sum(lowered.count(token) for token in sk_tokens)
    en_score = sum(lowered.count(token) for token in en_tokens)
    language = "sk" if sk_score >= en_score else "en"
    return {
        "language": language,
        "reason": f"{reason_prefix} Heuristika odhadla jazyk ako {language}.",
    }


def build_full_document_text(segmenty, max_chars=12000):
    parts = []
    total = 0
    for seg in segmenty:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        chunk = text[:remaining]
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)
