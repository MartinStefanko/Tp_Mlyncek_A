"""Uniform error format: {"error": {"code": ..., "message": ..., "details": {...}}}.

Every error leaves the API in this shape: our own ApiError, FastAPI request
validation errors, Starlette HTTP errors (unknown path, wrong method) and
unexpected exceptions (500 without exception details in the body).
"""

import logging
from typing import Any, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from qgen.contract import ApiErrorCode, ErrorBody, ErrorResponse

logger = logging.getLogger("qgen.api")


class ApiError(Exception):
    """An error the API returns to the caller."""

    def __init__(
        self,
        status_code: int,
        code: ApiErrorCode,
        message: str,
        details: Optional[dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        self.headers = dict(headers or {})


def error_response(
    status_code: int,
    code: ApiErrorCode,
    message: str,
    details: Optional[dict[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"), headers=dict(headers or {}))


def pydantic_error_list(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce Pydantic / FastAPI errors to loc, msg, type (no input values, no URLs)."""
    return [{"loc": list(err.get("loc", ())), "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))} for err in errors]


def invalid_options(message: str, errors: list[dict[str, Any]]) -> ApiError:
    return ApiError(400, "invalid_options", message, {"errors": errors})


def invalid_options_from(exc: ValidationError) -> ApiError:
    return invalid_options("Invalid options.", pydantic_error_list(exc.errors(include_url=False)))


_HTTP_STATUS_CODES: dict[int, ApiErrorCode] = {
    400: "validation_error",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    413: "file_too_large",
    415: "unsupported_file",
    422: "validation_error",
    429: "too_many_jobs",
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422, "validation_error", "Request does not match the schema.", {"errors": pydantic_error_list(list(exc.errors()))}
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_STATUS_CODES.get(exc.status_code, "internal_error")
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error."
        return error_response(exc.status_code, code, message, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unexpected error on %s %s: %s", request.method, request.url.path, type(exc).__name__, exc_info=exc)
        return error_response(500, "internal_error", "Internal server error.")
