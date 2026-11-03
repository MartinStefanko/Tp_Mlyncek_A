"""Checks of POST /v1/jobs input that do not depend on the backend: options and files."""

from typing import Optional

from fastapi import UploadFile
from pydantic import ValidationError

from qgen.contract import JobOptions, resolve_options
from service.backend import UploadedFile
from service.errors import ApiError, invalid_options, invalid_options_from
from service.settings import ServiceSettings

PDF_MAGIC = b"%PDF"
_READ_CHUNK = 1024 * 1024


def parse_options(raw: Optional[str], settings: ServiceSettings) -> JobOptions:
    """Parse the `options` form field (JSON string) into complete options.

    Any problem (bad JSON, wrong type, invalid combination, callback host not allowed)
    is 400 invalid_options with the list of errors in details.errors.
    """
    text = raw if raw is not None and raw.strip() else "{}"
    try:
        options = JobOptions.model_validate_json(text)
    except ValidationError as exc:
        raise invalid_options_from(exc) from None

    if options.callback_url is not None:
        check_callback_host(options, settings)

    try:
        return resolve_options(options, settings.option_defaults())
    except ValidationError as exc:  # misconfigured defaults
        raise invalid_options_from(exc) from None


def check_callback_host(options: JobOptions, settings: ServiceSettings) -> None:
    url = options.callback_url
    assert url is not None
    host = (url.host or "").lower()
    allowed = settings.allowed_webhook_hosts
    if host in allowed or f"{host}:{url.port}" in allowed:
        return
    reason = "webhooks are disabled (QGEN_WEBHOOK_ALLOWED_HOSTS is empty)" if not allowed else "host is not allowed"
    raise invalid_options(
        f"callback_url: {reason}.",
        [{"loc": ["callback_url"], "msg": f"Host '{host}' is not in QGEN_WEBHOOK_ALLOWED_HOSTS.", "type": "host_not_allowed"}],
    )


async def read_pdf_files(uploads: list[UploadFile], settings: ServiceSettings) -> list[UploadedFile]:
    """Check count, size and PDF header; return the file contents.

    Page count and password protection are not checked yet.
    """
    if len(uploads) > settings.max_files:
        raise ApiError(
            413,
            "file_too_large",
            f"Too many files: {len(uploads)} (limit {settings.max_files}).",
            {"max_files": settings.max_files, "files": len(uploads)},
        )

    limit = settings.max_upload_bytes
    result: list[UploadedFile] = []
    for index, upload in enumerate(uploads):
        name = upload.filename or f"file-{index + 1}.pdf"
        content = await _read_limited(upload, limit)
        if content is None:
            raise ApiError(
                413,
                "file_too_large",
                f"File '{name}' is larger than {settings.max_upload_mb:g} MB.",
                {"file": name, "max_upload_mb": settings.max_upload_mb},
            )
        if not content.startswith(PDF_MAGIC):
            raise ApiError(415, "unsupported_file", f"File '{name}' is not a PDF.", {"file": name})
        result.append(UploadedFile(filename=name, content=content))
    return result


async def _read_limited(upload: UploadFile, limit: int) -> Optional[bytes]:
    """Read the whole file, or return None as soon as it exceeds `limit` bytes."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)
