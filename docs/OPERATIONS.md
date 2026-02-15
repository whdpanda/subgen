# Operations

## Logs
- API process log (optional): `SUBGEN_LOG_PATH` (default: `/data/logs/api.log`)
- Per-job debug log: `/data/jobs/<job_id>/debug.log`

Trace id:
- HTTP requests: from `X-Request-Id` (or generated)
- Worker: trace id == job_id

## Metrics
`GET /metrics` exports:
- `subgen_http_requests_total{method,path,status}`
- `subgen_jobs_created_total{kind}`
- `subgen_jobs_status_read_total`
- `subgen_jobs_result_read_total`

## Common failure modes
- Path policy rejection: inputs must be under `SUBGEN_ALLOWED_ROOTS`
- Redis not reachable: API can create job spec but enqueue fails (should surface as 4xx/5xx per handler)
- Worker missing dependencies: job fails; inspect `/data/jobs/<job_id>/debug.log`