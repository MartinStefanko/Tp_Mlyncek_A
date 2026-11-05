# Real QGen service: the same image runs the API and the worker.
#   docker compose -f docker-compose.dev.yml up --build
# INSTALL_OCR=false (default here): no easyocr/torch; a job with extraction.ocr=true is
# rejected by the API. The full image with OCR (system libraries, models) is step 17.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

ARG INSTALL_OCR=false
COPY requirements-mock.txt requirements-service.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt \
 && if [ "$INSTALL_OCR" = "true" ]; then pip install --no-cache-dir -r requirements-ocr.txt; fi

COPY qgen/ qgen/
COPY service/ service/

# /data belongs to the app user, so a new named volume gets the same owner.
RUN useradd -m app && mkdir -p /data && chown app:app /data
USER app

ENV QGEN_BACKEND=real QGEN_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
