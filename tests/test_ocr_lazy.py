"""OCR is optional: easyocr is imported and its Reader built only when OCR is used."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

import qgen.extraction.text_extraction as text_extraction
from qgen.config import Settings
from qgen.extraction.text_extraction import OcrUnavailableError, get_ocr_reader, pdfs_to_text

REPO_ROOT = Path(__file__).resolve().parent.parent
EKO_PDF = Path(__file__).resolve().parent / "fixtures" / "eko.pdf"


def _run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )


def test_import_does_not_import_easyocr():
    result = _run_python(
        "import sys\n"
        "import qgen.extraction.text_extraction\n"
        "assert 'easyocr' not in sys.modules, 'easyocr imported at module import'\n"
        "assert 'cv2' not in sys.modules, 'cv2 imported at module import'\n"
    )
    assert result.returncode == 0, result.stderr


def test_import_works_without_easyocr():
    # sys.modules[name] = None makes `import easyocr` fail like a missing package.
    result = _run_python(
        "import sys\n"
        "sys.modules['easyocr'] = None\n"
        "import qgen.extraction.text_extraction as t\n"
        "assert t.ocr_available() is False\n"
        "segments = t.pdfs_to_text([r'%s'], ocr=False)\n"
        "assert segments\n" % EKO_PDF
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fresh_reader(monkeypatch):
    monkeypatch.setattr(text_extraction, "_ocr_reader", None)


def test_ocr_without_easyocr_raises_clear_error(monkeypatch, fresh_reader):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    with pytest.raises(OcrUnavailableError, match="easyocr is not installed"):
        pdfs_to_text([str(EKO_PDF)], ocr=True)
    with pytest.raises(OcrUnavailableError):
        get_ocr_reader()


def test_ocr_default_comes_from_settings(monkeypatch, fresh_reader):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    with pytest.raises(OcrUnavailableError):
        pdfs_to_text([str(EKO_PDF)], settings=Settings.model_construct(ocr_enabled=True))
    assert pdfs_to_text([str(EKO_PDF)], settings=Settings.model_construct(ocr_enabled=False))


def test_reader_is_created_once_on_first_use(monkeypatch, fresh_reader):
    created = []

    class FakeReader:
        def __init__(self, languages, gpu):
            created.append((languages, gpu))

    fake = types.ModuleType("easyocr")
    fake.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake)

    assert created == []
    first = get_ocr_reader()
    second = get_ocr_reader()
    assert first is second
    assert created == [(["sk", "en"], False)]


def test_no_ocr_does_not_create_reader(monkeypatch, fresh_reader):
    def fail():
        raise AssertionError("OCR reader must not be created when ocr=False")

    monkeypatch.setattr(text_extraction, "get_ocr_reader", fail)
    assert pdfs_to_text([str(EKO_PDF)], ocr=False)
    assert text_extraction._ocr_reader is None
