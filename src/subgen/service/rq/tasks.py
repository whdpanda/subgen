# src/subgen/service/rq/tasks.py
from __future__ import annotations

import os
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, cast

try:
    from rq import get_current_job
except ModuleNotFoundError:
    def get_current_job() -> None:
        return None

from subgen.agent.loop import LoopConfig, run_pr4c_closed_loop, safe_invoke_flat
from subgen.agent.tools import build_agent_tools
from subgen.agent.tools.tool_names import (
    BURN_SUBTITLES,
    FIX_SUBTITLES,
    QUALITY_CHECK_SUBTITLES,
    RUN_SUBGEN_PIPELINE,
)
from subgen.api.config import load_config
from subgen.api.schemas.jobs import (
    JobError,
    JobKind,
    JobResult,
    JobState,
    JobStatus,
    JobTimestamps,
    JobWorkerInfo,
    JobSpec,
)
from subgen.api.utils.logging_context import request_debug_logging
from subgen.core.jobs import JobSpecStore
from subgen.utils.logger import get_logger, set_trace_id, clear_trace_id

logger = get_logger("subgen.worker")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(cast(Any, x))
    return x


@lru_cache(maxsize=1)
def _tool_map() -> Dict[str, Any]:
    tools = build_agent_tools()
    tool_map = {t.name: t for t in tools}

    required = (RUN_SUBGEN_PIPELINE, QUALITY_CHECK_SUBTITLES, FIX_SUBTITLES, BURN_SUBTITLES)
    missing = [n for n in required if n not in tool_map]
    if missing:
        raise RuntimeError(f"Missing required tools: {missing}")

    return tool_map


def _loop_cfg_from_env() -> LoopConfig:
    cfg = LoopConfig()

    mp = os.getenv("SUBGEN_QUALITY_MAX_PASSES")
    if mp:
        try:
            cfg.max_passes = int(mp)
        except Exception:
            logger.debug("Invalid SUBGEN_QUALITY_MAX_PASSES=%s; keep default.", mp)

    emit = os.getenv("SUBGEN_EMIT_DEFAULT")
    if emit:
        cfg.emit_default = emit

    tgt = os.getenv("SUBGEN_TARGET_LANG_DEFAULT")
    if tgt:
        cfg.target_lang_default = tgt

    zhl = os.getenv("SUBGEN_ZH_LAYOUT_DEFAULT")
    if zhl:
        cfg.zh_layout_default = zhl.strip().lower() in ("1", "true", "yes", "y", "on")

    return cfg


def _cpu_safe_pipeline_defaults(pipeline_args: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pipeline_args or {})
    out.setdefault("asr_device", "cpu")
    out.setdefault("translator_device", "cpu")
    out.setdefault("demucs_device", "cpu")
    return out


def _write_status(store: JobSpecStore, job_id: str, status: JobStatus) -> None:
    store.write_status(job_id, status)


def _coerce_enum_by_name_or_value(enum_cls: Any, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except Exception:
            pass
        if value in enum_cls.__members__:
            return enum_cls[value]
    return value


def _load_spec(store: JobSpecStore, job_id: str) -> JobSpec:
    try:
        return store.read_as_model(job_id, JobSpec, "spec")
    except Exception:
        raw = store.read_spec_dict(job_id)
        raw = dict(raw or {})
        raw["kind"] = _coerce_enum_by_name_or_value(JobKind, raw.get("kind"))
        raw.setdefault("job_root", str(store.job_dir(job_id)))
        return JobSpec.model_validate(raw)


def _load_status_or_init(store: JobSpecStore, spec: JobSpec) -> JobStatus:
    if store.has_status(spec.job_id):
        try:
            return store.read_as_model(spec.job_id, JobStatus, "status")
        except Exception:
            raw = store.read_status_dict(spec.job_id)
            raw = dict(raw or {})
            raw["kind"] = _coerce_enum_by_name_or_value(JobKind, raw.get("kind", spec.kind))
            raw["state"] = _coerce_enum_by_name_or_value(JobState, raw.get("state", JobState.QUEUED))
            raw.setdefault("timestamps", {"created_at": spec.created_at})
            raw.setdefault("worker", {"worker_id": None, "attempt": 1})
            return JobStatus.model_validate(raw)

    return JobStatus(
        job_id=spec.job_id,
        kind=spec.kind,
        state=JobState.QUEUED,
        progress=None,
        timestamps=JobTimestamps(created_at=spec.created_at),
        worker=JobWorkerInfo(worker_id=None, attempt=1),
        message="queued",
        error=None,
        artifacts=None,
    )


def _mark_started(store: JobSpecStore, status: JobStatus, worker_id: str) -> JobStatus:
    status.state = JobState.STARTED  # type: ignore[misc]
    status.message = "started"  # type: ignore[misc]
    status.worker.worker_id = worker_id  # type: ignore[misc]
    if status.timestamps.started_at is None:
        status.timestamps.started_at = _utc_now()  # type: ignore[misc]
    _write_status(store, status.job_id, status)
    return status


def _mark_running(store: JobSpecStore, status: JobStatus, progress: Optional[float], message: str) -> JobStatus:
    status.state = JobState.RUNNING  # type: ignore[misc]
    status.progress = progress  # type: ignore[misc]
    status.message = message  # type: ignore[misc]
    _write_status(store, status.job_id, status)
    return status


def _mark_succeeded(store: JobSpecStore, status: JobStatus, result: JobResult) -> None:
    status.state = JobState.SUCCEEDED  # type: ignore[misc]
    status.progress = 1.0  # type: ignore[misc]
    status.message = "succeeded"  # type: ignore[misc]
    if status.timestamps.finished_at is None:
        status.timestamps.finished_at = _utc_now()  # type: ignore[misc]
    status.error = None  # type: ignore[misc]
    _write_status(store, status.job_id, status)
    store.write_result(status.job_id, result)


def _mark_failed(store: JobSpecStore, status: JobStatus, code: str, detail: str) -> None:
    status.state = JobState.FAILED  # type: ignore[misc]
    status.progress = None  # type: ignore[misc]
    status.message = "failed"  # type: ignore[misc]
    if status.timestamps.finished_at is None:
        status.timestamps.finished_at = _utc_now()  # type: ignore[misc]
    status.error = JobError(code=code, detail=detail)  # type: ignore[misc]
    _write_status(store, status.job_id, status)

    store.write_result(
        status.job_id,
        JobResult(
            job_id=status.job_id,
            kind=status.kind,
            ok=False,
            primary_path=None,
            srt_paths=None,
            outputs={},
            artifacts={},
            meta={"error": {"code": code, "detail": detail}},
        ),
    )


def _run_generate(spec: JobSpec, job_dir: str) -> Dict[str, Any]:
    tool_map = _tool_map()
    loop_cfg = _loop_cfg_from_env()

    inputs = spec.inputs or {}
    video_path = inputs["video_path"]
    out_dir = inputs["out_dir"]

    max_passes = inputs.get("max_passes")
    pipeline_args = inputs.get("pipeline_args") or {}
    quality_args = inputs.get("quality_args")
    fix_args = inputs.get("fix_args")

    pipeline_args = _cpu_safe_pipeline_defaults(dict(pipeline_args))
    pipeline_args.setdefault("job_id", spec.job_id)

    p_args: Dict[str, Any] = {"video_path": video_path, "out_dir": out_dir}
    p_args.update(pipeline_args)

    with request_debug_logging(out_dir=job_dir, capture_stdout=True, logger_name="subgen"):
        res = run_pr4c_closed_loop(
            tool_map,
            pipeline_args=p_args,
            cfg=loop_cfg,
            max_passes=max_passes,
            quality_args=quality_args,
            fix_args=fix_args,
            burn_args=None,
        )

    out = _as_jsonable(res)
    assert isinstance(out, dict)
    return out


def _run_fix(spec: JobSpec, job_dir: str) -> Dict[str, Any]:
    tool_map = _tool_map()
    inputs = spec.inputs or {}

    srt_path = inputs["srt_path"]
    out_dir = inputs.get("out_dir")
    fix_args = inputs.get("fix_args")
    quality_args = inputs.get("quality_args")

    payload: Dict[str, Any] = {"srt_path": srt_path}
    if out_dir:
        payload["out_dir"] = out_dir
    if fix_args:
        payload.update(fix_args)

    with request_debug_logging(out_dir=job_dir, capture_stdout=True, logger_name="subgen"):
        fix_flat = safe_invoke_flat(tool_map[FIX_SUBTITLES], FIX_SUBTITLES, payload)

        fixed_path = fix_flat.get("fixed_srt_path")
        final_srt = fixed_path if isinstance(fixed_path, str) and fixed_path else srt_path

        q_flat: Optional[Dict[str, Any]] = None
        report_path: Optional[str] = None
        try:
            q_payload: Dict[str, Any] = {"srt_path": final_srt}
            if out_dir:
                q_payload["out_dir"] = out_dir
            if quality_args:
                q_payload.update(quality_args)

            q_flat = safe_invoke_flat(tool_map[QUALITY_CHECK_SUBTITLES], QUALITY_CHECK_SUBTITLES, q_payload)
            rp = q_flat.get("report_path")
            report_path = rp if isinstance(rp, str) and rp else None
        except Exception:
            logger.exception("fix: post-quality check failed (best-effort).")
            q_flat = None
            report_path = None

    ok_fix = bool(fix_flat.get("ok"))
    ok_quality = bool(q_flat.get("ok")) if isinstance(q_flat, dict) else True

    meta: Dict[str, Any] = {
        "input_srt_path": srt_path,
        "final_srt_path": final_srt,
        "report_path": report_path,
    }
    if isinstance(fix_flat.get("meta"), dict) and fix_flat["meta"].get("error"):
        meta["error"] = fix_flat["meta"]["error"]
    if isinstance(q_flat, dict) and isinstance(q_flat.get("meta"), dict) and q_flat["meta"].get("error"):
        meta.setdefault("errors", []).append(q_flat["meta"]["error"])

    return {
        "ok": bool(ok_fix and ok_quality),
        "srt_path": final_srt,
        "report_path": report_path,
        "meta": meta,
        "fix": fix_flat,
        "quality": q_flat,
    }


def _run_burn(spec: JobSpec, job_dir: str) -> Dict[str, Any]:
    tool_map = _tool_map()
    inputs = spec.inputs or {}

    video_path = inputs["video_path"]
    srt_path = inputs["srt_path"]
    out_path = inputs.get("out_path")
    burn_args = inputs.get("burn_args")

    payload: Dict[str, Any] = {"video_path": video_path, "srt_path": srt_path}
    if out_path:
        payload["out_path"] = out_path
    if burn_args:
        payload.update(burn_args)

    with request_debug_logging(out_dir=job_dir, capture_stdout=True, logger_name="subgen"):
        burn_flat = safe_invoke_flat(tool_map[BURN_SUBTITLES], BURN_SUBTITLES, payload)

    return burn_flat


def run_job(job_id: str) -> Dict[str, Any]:
    """
    RQ task entrypoint:
      "subgen.service.rq.tasks.run_job"
    """
    # Correlate all worker logs by job_id
    set_trace_id(job_id)

    cfg = load_config()
    store = JobSpecStore(cfg.job_root)

    worker_id = os.getenv("HOSTNAME") or os.getenv("COMPUTERNAME") or "worker"
    try:
        _ = get_current_job()
    except Exception:
        pass

    spec = _load_spec(store, job_id)
    status = _load_status_or_init(store, spec)

    status = _mark_started(store, status, worker_id=worker_id)

    job_dir = str(Path(cfg.job_root) / job_id)
    try:
        logger.info("JOB_START job_id=%s kind=%s job_dir=%s", job_id, spec.kind.value, job_dir)
        _mark_running(store, status, progress=0.05, message="running")

        if spec.kind == JobKind.SUBTITLES_GENERATE:
            out = _run_generate(spec, job_dir=job_dir)
            ok = bool(out.get("ok", True))
            primary_path = out.get("primary_path") or out.get("primary") or out.get("primaryPath")
            srt_paths = out.get("srt_paths")

            result = JobResult(
                job_id=job_id,
                kind=spec.kind,
                ok=ok,
                primary_path=primary_path if isinstance(primary_path, str) else None,
                srt_paths=srt_paths if isinstance(srt_paths, list) else None,
                outputs=out.get("outputs") if isinstance(out.get("outputs"), dict) else {},
                artifacts=out.get("artifacts") if isinstance(out.get("artifacts"), dict) else {"raw": out},
                meta=out.get("meta") if isinstance(out.get("meta"), dict) else {},
            )

        elif spec.kind == JobKind.SUBTITLES_FIX:
            out = _run_fix(spec, job_dir=job_dir)
            ok = bool(out.get("ok", False))
            final_srt = out.get("srt_path")
            report_path = out.get("report_path")

            result = JobResult(
                job_id=job_id,
                kind=spec.kind,
                ok=ok,
                primary_path=final_srt if isinstance(final_srt, str) else None,
                srt_paths=[final_srt] if isinstance(final_srt, str) else None,
                outputs={"srt_path": final_srt, "report_path": report_path},
                artifacts={"raw": out},
                meta=out.get("meta") if isinstance(out.get("meta"), dict) else {},
            )

        elif spec.kind == JobKind.SUBTITLES_BURN:
            out = _run_burn(spec, job_dir=job_dir)
            ok = bool(out.get("ok", False))
            out_path = out.get("out_path") or out.get("output_path") or out.get("path")

            result = JobResult(
                job_id=job_id,
                kind=spec.kind,
                ok=ok,
                primary_path=out_path if isinstance(out_path, str) else None,
                srt_paths=None,
                outputs=out if isinstance(out, dict) else {"raw": out},
                artifacts={},
                meta={},
            )

        else:
            raise ValueError(f"Unknown job kind: {spec.kind}")

        _mark_running(store, status, progress=0.95, message="finalizing")
        _mark_succeeded(store, status, result)

        logger.info("JOB_DONE job_id=%s kind=%s ok=%s", job_id, spec.kind.value, result.ok)
        return {"ok": True, "job_id": job_id, "kind": spec.kind.value, "primary_path": result.primary_path}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("JOB_FAILED job_id=%s kind=%s err=%s", job_id, getattr(spec.kind, "value", spec.kind), e)
        _mark_failed(store, status, code="JOB_FAILED", detail=f"{e}\n{tb}")
        raise
    finally:
        clear_trace_id()