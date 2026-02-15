# Async Jobs API

## Overview
The legacy `/v1/subtitles/*` endpoints now return `202 Accepted` with a `JobStatus` payload (async job).

Additionally, `/v1/jobs/*` offers explicit job endpoints.

## Job lifecycle
- queued -> started/running -> succeeded|failed

Disk is the source of truth:
- `status.json` is updated by API/worker
- `result.json` is written by worker

## Endpoints

### Create jobs (explicit)
- `POST /v1/jobs/subtitles/generate`
- `POST /v1/jobs/subtitles/fix`
- `POST /v1/jobs/subtitles/burn`

Response includes:
- `job_id`
- `status_url`
- `result_url`

### Poll
- `GET /v1/jobs/{job_id}` -> status
- `GET /v1/jobs/{job_id}/result` -> result (404 until ready)

### Legacy routes (now async)
- `POST /v1/subtitles/generate` -> 202 + JobStatus
- `POST /v1/subtitles/fix` -> 202 + JobStatus
- `POST /v1/subtitles/burn` -> 202 + JobStatus

## Observability
- Access logs include trace id.
- Prometheus endpoint: `GET /metrics`
