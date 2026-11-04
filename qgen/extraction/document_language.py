from typing import Optional

from qgen.config import Settings, setting_or_default
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_LANGUAGE, ProgressCallback, report_progress


SUPPORTED_DOCUMENT_LANGUAGES = {"sk", "en"}
LANGUAGE_FROM_OPTION_REASON = "Language set by the 'language' option; detection skipped."


def detect_document_language(
    segmenty, llm=None, model=None, verbose=True, max_chars=None, role="lo", *, settings: Settings
):
    model = setting_or_default(model, "llm_model_lo", settings)
    source_text = build_full_document_text(segmenty, max_chars=max_chars)
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

    # Step 6: no `except Exception` with the heuristic fallback any more. A failed call or
    # invalid JSON raises LLMError, which must reach run_pipeline (job failed, llm_error).
    # The heuristic stays only for a valid JSON answer with a wrong shape or language.
    parsed = generate_json(llm, prompt, role=role, model=model, verbose=verbose, settings=settings)

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


def build_full_document_text(segmenty, max_chars=None):
    max_chars = setting_or_default(max_chars, "language_sample_max_chars")
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


def resolve_document_language(
    segmenty,
    document_language: Optional[str],
    llm,
    model: str,
    settings: Settings,
    verbose: bool,
    progress_callback: Optional[ProgressCallback],
    role: str = "lo",
) -> dict:
    """Language of the document: from the option (sk/en) or detected by the LLM (None/auto)."""
    report_progress(progress_callback, STAGE_LANGUAGE, 0, 1)
    if document_language in (None, "auto"):
        language_info = detect_document_language(
            segmenty,
            llm=llm,
            model=model,
            verbose=verbose,
            max_chars=settings.language_sample_max_chars,
            role=role,
            settings=settings,
        )
    elif document_language in SUPPORTED_DOCUMENT_LANGUAGES:
        language_info = {"language": document_language, "reason": LANGUAGE_FROM_OPTION_REASON}
    else:
        raise ValueError(f"Unsupported document language: {document_language!r} (allowed: auto, sk, en).")
    report_progress(progress_callback, STAGE_LANGUAGE, 1, 1)
    return language_info
