import importlib.util
import os
import threading

import fitz
import numpy as np
from pypdf import PdfReader

from qgen.config import get_settings
from qgen.progress import STAGE_EXTRACT, report_progress

ZNAKY = "ˇ´`^~˝˚’'"
OCR_LANGUAGES = ['sk', 'en']


class OcrUnavailableError(RuntimeError):
    """OCR was requested, but easyocr is not installed (image built without OCR)."""


_ocr_reader = None
_ocr_lock = threading.Lock()


def ocr_available() -> bool:
    return importlib.util.find_spec("easyocr") is not None


def ensure_ocr_available() -> None:
    if not ocr_available():
        raise OcrUnavailableError(
            "OCR was requested (ocr=true), but easyocr is not installed. "
            "Install easyocr or run without OCR."
        )


def get_ocr_reader():
    """easyocr.Reader, created on first use, once per process."""
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                try:
                    import easyocr
                except ImportError as e:
                    raise OcrUnavailableError(
                        "OCR was requested (ocr=true), but easyocr is not installed. "
                        "Install easyocr or run without OCR."
                    ) from e
                _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def ocr_image(img_bytes):
    import cv2

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = get_ocr_reader().readtext(img, detail=0, paragraph=True)
    return "\n".join(result).strip()


def pdf_to_text(pdf_cesta: str, source_id=None, source_name=None, ocr=False, on_page=None):
    reader_pdf = PdfReader(pdf_cesta)
    doc_fitz = fitz.open(pdf_cesta) if ocr else None
    source_name = source_name or os.path.basename(pdf_cesta)

    segmenty = []

    for i, page in enumerate(reader_pdf.pages, start=1):
        text = page.extract_text() or ""
        text = text.translate(str.maketrans("", "", ZNAKY)).strip()

        if source_id:
            blok = f"Dokument {source_id}: {source_name}\nStrana {i}\n{text}"
        else:
            blok = f"Strana {i}\n{text}"

        if ocr:
            page_fitz = doc_fitz[i - 1]
            images = page_fitz.get_images(full=True)

            for idx, img_info in enumerate(images, start=1):
                xref = img_info[0]
                base_image = doc_fitz.extract_image(xref)
                ocr_text = ocr_image(base_image["image"])

                if ocr_text:
                    blok += f"\n\n[OCR OBRÁZOK {idx}]\n{ocr_text}"

        segmenty.append(
            {
                "page": i,
                "source_id": source_id,
                "source_name": source_name,
                "text": blok,
            }
        )
        if on_page is not None:
            on_page()

    return segmenty


def pdfs_to_text(pdf_subory, ocr=None, progress_callback=None, settings=None):
    """Extract pages of all PDFs. ocr=None means QGEN_OCR_ENABLED (default false)."""
    if ocr is None:
        ocr = (settings or get_settings()).ocr_enabled
    if ocr:
        ensure_ocr_available()

    pdf_list = []
    for pdf_info in pdf_subory:
        if isinstance(pdf_info, dict):
            pdf_list.append((pdf_info.get("path"), pdf_info.get("name")))
        else:
            pdf_cesta = str(pdf_info)
            pdf_list.append((pdf_cesta, os.path.basename(pdf_cesta)))

    total_pages = 0
    if progress_callback is not None:
        total_pages = sum(len(PdfReader(pdf_cesta).pages) for pdf_cesta, _ in pdf_list)
    done = 0
    report_progress(progress_callback, STAGE_EXTRACT, done, total_pages)

    def page_done():
        nonlocal done
        done += 1
        report_progress(progress_callback, STAGE_EXTRACT, done, total_pages)

    segmenty = []
    for index, (pdf_cesta, source_name) in enumerate(pdf_list, start=1):
        source_id = f"D{index}"
        segmenty.extend(
            pdf_to_text(
                pdf_cesta,
                source_id=source_id,
                source_name=source_name,
                ocr=ocr,
                on_page=page_done,
            )
        )
    return segmenty
