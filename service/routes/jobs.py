"""Job endpoints (Príloha A). Routes check the request and call the JobBackend."""

from typing import Annotated, Any, Optional, Union

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from qgen.contract import CreateJobResponse, ErrorResponse, JobLinks, JobListResponse, JobStatus, Report, UnitsResult
from service.backend import CreateJobRequest
from service.intake import check_ocr, inspect_pdfs, parse_options, read_pdf_files
from service.routes.deps import Backend, Owner, Settings

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _errors(*codes: tuple[int, str]) -> dict[int | str, dict[str, Any]]:
    result: dict[int | str, dict[str, Any]] = {401: {"model": ErrorResponse, "description": "unauthorized"}}
    for status, description in codes:
        result[status] = {"model": ErrorResponse, "description": description}
    return result


def _uploads(values: Optional[list[Union[UploadFile, str]]], field: str) -> list[UploadFile]:
    """The files of one form field. Empty text parts are skipped; other text is a 422 as before."""
    found: list[UploadFile] = []
    for index, value in enumerate(values or []):
        if isinstance(value, StarletteUploadFile):
            found.append(value)
        elif isinstance(value, str) and value.strip() == "":
            continue
        else:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("body", field, index),
                        "msg": f"Value error, Expected UploadFile, received: {type(value)}",
                        "input": None,
                    }
                ]
            )
    return found


def _links(job_id: str) -> JobLinks:
    return JobLinks(self_=f"/v1/jobs/{job_id}")


@router.post(
    "",
    status_code=202,
    response_model=CreateJobResponse,
    summary="Create a job: multipart files[] (PDF) + options (JSON string).",
    responses={
        200: {"model": CreateJobResponse, "description": "external_ref already exists: the original job, replayed=true."},
        **_errors(
            (400, "invalid_options"),
            (413, "file_too_large (also more files than QGEN_MAX_FILES)"),
            (415, "unsupported_file"),
            (422, "validation_error (e.g. no files)"),
            (429, "too_many_jobs (header Retry-After)"),
        ),
    },
)
async def create_job(
    response: Response,
    backend: Backend,
    settings: Settings,
    owner: Owner,
    # Union with str: Swagger UI sends an empty text part for an empty file field
    # (-F 'files[]='); such parts are ignored as if the field were missing (_uploads).
    files: Annotated[
        Optional[list[Union[UploadFile, str]]], File(description="PDF files (field name 'files').")
    ] = None,
    files_brackets: Annotated[
        Optional[list[Union[UploadFile, str]]],
        File(alias="files[]", description="PDF files (field name 'files[]', e.g. from Laravel)."),
    ] = None,
    options: Annotated[Optional[str], Form(description="JSON string with options; all fields optional.")] = None,
) -> CreateJobResponse:
    uploads = [*_uploads(files, "files"), *_uploads(files_brackets, "files[]")]
    if not uploads:
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body", "files"), "msg": "Field required", "input": None}]
        )
    # Order of the checks: kap. 3 – Význam odpovede 202 (API key is the Owner dependency).
    resolved = parse_options(options, settings)
    if settings.backend == "real":
        check_ocr(resolved)
    pdfs = await read_pdf_files(uploads, settings)
    if settings.backend == "real":
        pdfs = await run_in_threadpool(inspect_pdfs, pdfs, settings)
    created = await run_in_threadpool(backend.create, CreateJobRequest(owner=owner, options=resolved, files=pdfs))
    if created.replayed:
        response.status_code = 200
    status = created.status
    return CreateJobResponse(job_id=status.job_id, status=status.status, replayed=created.replayed, links=_links(status.job_id))


@router.get(
    "",
    response_model=JobListResponse,
    summary="Jobs with the given external_ref, newest first.",
    responses=_errors((422, "validation_error (missing external_ref)")),
)
async def list_jobs(
    backend: Backend,
    owner: Owner,
    external_ref: Annotated[str, Query(min_length=1, max_length=255)],
) -> JobListResponse:
    items = await run_in_threadpool(backend.list_by_external_ref, external_ref, owner)
    return JobListResponse(jobs=items)


@router.get("/{job_id}", response_model=JobStatus, summary="Job state.", responses=_errors((404, "job_not_found")))
async def get_job(job_id: str, backend: Backend, owner: Owner) -> JobStatus:
    return await run_in_threadpool(backend.get, job_id, owner)


@router.get(
    "/{job_id}/result",
    response_model=UnitsResult,
    summary="Result exactly in the units format.",
    responses=_errors(
        (404, "job_not_found"),
        (409, "result_not_ready (queued/running) or job_not_completed (failed/cancelled)"),
    ),
)
async def get_result(job_id: str, backend: Backend, owner: Owner) -> UnitsResult:
    return await run_in_threadpool(backend.result, job_id, owner)


@router.get(
    "/{job_id}/report",
    response_model=Report,
    summary="Sections, question metadata, quality, warnings, timings.",
    responses=_errors(
        (404, "job_not_found"),
        (409, "result_not_ready (queued/running) or job_not_completed (failed/cancelled)"),
    ),
)
async def get_report(job_id: str, backend: Backend, owner: Owner) -> Report:
    return await run_in_threadpool(backend.report, job_id, owner)


@router.delete(
    "/{job_id}",
    response_model=JobStatus,
    summary="Cancel a queued/running job (200, status cancelled) or delete a finished one (204).",
    responses={204: {"description": "Job, result and report deleted."}, **_errors((404, "job_not_found"))},
)
async def delete_job(job_id: str, backend: Backend, owner: Owner) -> Any:
    status = await run_in_threadpool(backend.cancel_or_delete, job_id, owner)
    if status is None:
        return Response(status_code=204)
    return status


@router.post(
    "/{job_id}/retry",
    status_code=202,
    response_model=JobStatus,
    summary="Run a failed job again with the same job_id (within 24 h of the failure).",
    responses=_errors((404, "job_not_found"), (409, "job_not_retryable"), (429, "too_many_jobs (header Retry-After)")),
)
async def retry_job(job_id: str, backend: Backend, owner: Owner) -> JobStatus:
    return await run_in_threadpool(backend.retry, job_id, owner)
