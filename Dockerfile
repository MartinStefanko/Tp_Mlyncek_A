# QGen service: one image for api, worker and webhook-worker (docker-compose.yml).
#   docker compose up -d --build
# Smaller image without OCR: QGEN_INSTALL_OCR=false in .env (or
#   docker compose build --build-arg INSTALL_OCR=false); there extraction.ocr=true -> 400 invalid_options.
# OCR: torch only for CPU (versions from requirements-ocr.txt), opencv-python-headless,
# EasyOCR models are downloaded here into /models (EASYOCR_MODULE_PATH), never at run time.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EASYOCR_MODULE_PATH=/models
WORKDIR /app

ARG INSTALL_OCR=true
COPY requirements-mock.txt requirements-service.txt requirements-ocr.txt ./
RUN pip install -r requirements-service.txt \
 && if [ "$INSTALL_OCR" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu \
          $(grep -E '^(torch|torchvision)==' requirements-ocr.txt) \
   && pip install -r requirements-ocr.txt \
   && python -c "import easyocr; easyocr.Reader(['sk', 'en'], gpu=False, verbose=False)"; \
    fi

COPY qgen/ qgen/
COPY service/ service/

# /data belongs to the app user, so a new named volume gets the same owner.
RUN useradd --create-home --uid 1000 app && mkdir -p /data /models && chown app:app /data
USER app

ENV QGEN_BACKEND=real QGEN_DATA_DIR=/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "service.healthcheck", "api"]
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
