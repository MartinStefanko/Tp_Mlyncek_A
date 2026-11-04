"""Step 6 guards: vendor SDKs only in qgen/llm/adapters/, raw client calls only in qgen/llm/."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHECKED_DIRS = ("qgen", "service", "tools", "tests")
ADAPTERS = REPO / "qgen" / "llm" / "adapters"
LLM_PACKAGE = REPO / "qgen" / "llm"

VENDOR_IMPORT = re.compile(r"^\s*(?:import|from)\s+(google|openai|anthropic)\b", re.M)
RAW_CLIENT_CALL = re.compile(r"\.models\.(?:generate_content|embed_content)\s*\(")


def _python_files():
    for name in CHECKED_DIRS:
        for path in (REPO / name).rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def _inside(path: Path, folder: Path) -> bool:
    return folder in path.parents


def test_vendor_libraries_are_imported_only_in_adapters():
    offenders = [
        str(path.relative_to(REPO))
        for path in _python_files()
        if not _inside(path, ADAPTERS) and VENDOR_IMPORT.search(path.read_text(encoding="utf-8-sig"))
    ]
    assert offenders == []


def test_google_is_imported_in_the_gemini_adapter():
    assert VENDOR_IMPORT.search((ADAPTERS / "gemini.py").read_text(encoding="utf-8"))


def test_no_raw_client_calls_outside_llm_package():
    offenders = [
        str(path.relative_to(REPO))
        for path in _python_files()
        if not _inside(path, LLM_PACKAGE)
        and path.name != Path(__file__).name
        and RAW_CLIENT_CALL.search(path.read_text(encoding="utf-8-sig"))
    ]
    assert offenders == []
