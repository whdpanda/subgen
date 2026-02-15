from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from subgen.api.config import load_config
from subgen.api.metrics import inc_job_created, inc_job_result_read, inc_job_status_read
from subgen.api.schemas.jobs import (
    BurnJobRequest,
    FixJobRequest,
    GenerateJobRequest,
    JobCreateResponse,
    JobResultResponse,
    JobStatusResponse,
)
from subgen.api.schemas.jobs import JobKind
from subgen.api.services.jobs_service import JobsService
from subgen.core.jobs import JobSpecStore
from subgen.service.rq.conn import get_redis_connection
from subgen.service.rq.queues import get_queue

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


def _urls(job_id: str) -> tuple[str, str]:
    return (f"/v1/jobs/{job_id}", f"/v1/jobs/{job_id}/result")


@lru_cache(maxsize=1)
def _svc() -> JobsService:
    cfg = load_config()
    store = JobSpecStore(cfg.job_root)
    conn = get_redis_connection(cfg.redis_url)
    q = get_queue(conn, cfg.rq_queue_name, cfg.rq_job_timeout_sec)
    return JobsService(cfg=cfg, store=store, queue=q)


@router.post("/subtitles/generate", response_model=JobCreateResponse)
def create_generate_job(
    req: GenerateJobRequest,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
) -> JobCreateResponse:
    try:
        r = _svc().create_and_enqueue(
            kind=JobKind.SUBTITLES_GENERATE,
            inputs=req.model_dump(),
            meta={"client_request_id": x_request_id} if x_request_id else {},
        )
        inc_job_created(kind=JobKind.SUBTITLES_GENERATE.value)
        status_url, result_url = _urls(r.spec.job_id)
        return JobCreateResponse(job_id=r.spec.job_id, status_url=status_url, result_url=result_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/subtitles/fix", response_model=JobCreateResponse)
def create_fix_job(
    req: FixJobRequest,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
) -> JobCreateResponse:
    try:
        r = _svc().create_and_enqueue(
            kind=JobKind.SUBTITLES_FIX,
            inputs=req.model_dump(),
            meta={"client_request_id": x_request_id} if x_request_id else {},
        )
        inc_job_created(kind=JobKind.SUBTITLES_FIX.value)
        status_url, result_url = _urls(r.spec.job_id)
        return JobCreateResponse(job_id=r.spec.job_id, status_url=status_url, result_url=result_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/subtitles/burn", response_model=JobCreateResponse)
def create_burn_job(
    req: BurnJobRequest,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
) -> JobCreateResponse:
    try:
        r = _svc().create_and_enqueue(
            kind=JobKind.SUBTITLES_BURN,
            inputs=req.model_dump(),
            meta={"client_request_id": x_request_id} if x_request_id else {},
        )
        inc_job_created(kind=JobKind.SUBTITLES_BURN.value)
        status_url, result_url = _urls(r.spec.job_id)
        return JobCreateResponse(job_id=r.spec.job_id, status_url=status_url, result_url=result_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        st = _svc().get_status(job_id)
        inc_job_status_read()
        status_url, result_url = _urls(job_id)
        return JobStatusResponse(job=st, status_url=status_url, result_url=result_url)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str) -> JobResultResponse:
    try:
        res = _svc().get_result(job_id)
        inc_job_result_read()
        return JobResultResponse(job_id=job_id, kind=res.kind, result=res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e