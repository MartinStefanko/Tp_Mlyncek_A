"""Manual smoke test of the configured LLM and embedding providers (step 7).

Makes ONE real generate call and ONE real embed call with the current configuration
(.env / environment), without the retry layer, and prints provider, model, time and usage.
For checking a provider by hand; it costs a few tokens and is not run in CI or by pytest.

Usage (from the repo root):
    venv/Scripts/python.exe tools/smoke_llm.py [--role lo|prereq|questions|eval|check|topics|split]
                                               [--skip-llm] [--skip-embed]

Exit code: 0 all calls succeeded, 1 a call failed, 2 invalid configuration.
Keys are never printed. The prompt is a fixed test sentence, not a document.
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qgen.config import Settings, get_settings  # noqa: E402
from qgen.llm import factory  # noqa: E402
from qgen.llm.errors import LLMConfigError, ProviderError  # noqa: E402
from qgen.llm.json_load import safe_load_json  # noqa: E402

PROMPT = (
    'Return ONLY valid JSON in this format: {"language": "sk" or "en", "greeting": "<the greeting>"}.\n'
    "Text: Dobrý deň, vitajte na prednáške."
)
EMBED_TEXTS = ["Fotosyntéza je proces v rastlinách.", "Photosynthesis is a process in plants."]


def _llm_target(settings: Settings) -> str:
    if settings.llm_provider == "openai_compatible":
        return f"openai_compatible @ {settings.llm_base_url}"
    return settings.llm_provider


def _embed_target(settings: Settings) -> str:
    if settings.embed_provider == "openai_compatible":
        return f"openai_compatible @ {settings.embed_base_url}"
    return settings.embed_provider


def _error_text(error: Exception) -> str:
    # ProviderError messages are ours (no response body); for others only the type.
    if isinstance(error, ProviderError):
        return f"{type(error).__name__}: {error} (status={error.status}, retryable={error.retryable})"
    return type(error).__name__


def smoke_generate(settings: Settings, role: str) -> bool:
    model = factory.model_for_role(role, settings)
    print(f"[generate] provider={_llm_target(settings)} role={role} model={model}")
    llm = factory.get_llm(role, settings)
    start = time.perf_counter()
    try:
        response = llm.generate(PROMPT, model=model, json_mode=True)
    except Exception as error:
        print(f"[generate] FAILED after {time.perf_counter() - start:.2f}s: {_error_text(error)}")
        return False
    seconds = time.perf_counter() - start
    print(f"[generate] ok in {seconds:.2f}s, answered by {response.model}, usage={response.usage or '-'}")
    preview = response.text.strip().replace("\n", " ")
    print(f"[generate] answer: {preview[:200]}{'...' if len(preview) > 200 else ''}")
    try:
        parsed = safe_load_json(response.text)
        print(f"[generate] JSON ok: {type(parsed).__name__}")
    except ValueError:
        print("[generate] WARNING: the answer is not valid JSON (the pipeline would repeat the call once)")
    return True


def smoke_embed(settings: Settings) -> bool:
    model = settings.embed_model
    print(f"[embed]    provider={_embed_target(settings)} model={model}")
    embedder = factory.get_embedder(settings)
    start = time.perf_counter()
    try:
        vectors = embedder.embed(EMBED_TEXTS, model=model)
    except Exception as error:
        print(f"[embed]    FAILED after {time.perf_counter() - start:.2f}s: {_error_text(error)}")
        return False
    seconds = time.perf_counter() - start
    dims = sorted({len(v) for v in vectors})
    print(f"[embed]    ok in {seconds:.2f}s, {len(vectors)} vectors, dimension {dims}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="One real generate and embed call with the current configuration.")
    parser.add_argument("--role", choices=factory.ROLES, default="lo", help="Role whose model is used (default: lo).")
    parser.add_argument("--skip-llm", action="store_true", help="Do not call generate.")
    parser.add_argument("--skip-embed", action="store_true", help="Do not call embed.")
    args = parser.parse_args(argv)

    settings = get_settings()
    try:
        factory.check_config(settings, llm=not args.skip_llm, embedder=not args.skip_embed)
    except LLMConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    ok = True
    if not args.skip_llm:
        ok = smoke_generate(settings, args.role) and ok
    if not args.skip_embed:
        ok = smoke_embed(settings) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
