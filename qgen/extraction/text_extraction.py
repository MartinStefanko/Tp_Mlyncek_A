from pypdf import PdfReader
import fitz
import easyocr
import numpy as np
import cv2
import os

ZNAKY = "ˇ´`^~˝˚’'"
reader_ocr = easyocr.Reader(['sk', 'en'], gpu=False)

def pdf_to_text(pdf_cesta: str, source_id=None, source_name=None):
    reader_pdf = PdfReader(pdf_cesta)
    doc_fitz = fitz.open(pdf_cesta)
    source_name = source_name or os.path.basename(pdf_cesta)

    segmenty = []

    for i, page in enumerate(reader_pdf.pages, start=1):
        text = page.extract_text() or ""
        text = text.translate(str.maketrans("", "", ZNAKY)).strip()

        if source_id:
            blok = f"Dokument {source_id}: {source_name}\nStrana {i}\n{text}"
        else:
            blok = f"Strana {i}\n{text}"

        page_fitz = doc_fitz[i - 1]
        images = page_fitz.get_images(full=True)

        for idx, img_info in enumerate(images, start=1):
            xref = img_info[0]
            base_image = doc_fitz.extract_image(xref)
            img_bytes = base_image["image"]

            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            result = reader_ocr.readtext(img, detail=0, paragraph=True)
            ocr_text = "\n".join(result).strip()

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


    return segmenty


def pdfs_to_text(pdf_subory):
    segmenty = []
    for index, pdf_info in enumerate(pdf_subory, start=1):
        if isinstance(pdf_info, dict):
            pdf_cesta = pdf_info.get("path")
            source_name = pdf_info.get("name")
        else:
            pdf_cesta = str(pdf_info)
            source_name = os.path.basename(pdf_cesta)

        source_id = f"D{index}"
        segmenty.extend(
            pdf_to_text(
                pdf_cesta,
                source_id=source_id,
                source_name=source_name,
            )
        )
    return segmenty
