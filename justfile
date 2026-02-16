# =========================
# PR6 Acceptance justfile
# =========================

set shell := ["C:/Program Files/Git/bin/bash.exe", "-eu", "-o", "pipefail", "-c"]

# ---- Config (override via: `just VAR=value target`) ----
IMAGE        := "subgen:pr6"

REDIS_NAME   := "subgen-redis"
API_NAME     := "subgen-api"
WORKER_NAME  := "subgen-worker"

HOST         := "127.0.0.1"
PORT         := "8000"
BASE_URL     := "http://" + HOST + ":" + PORT

# ---- Host paths (relative; created under repo) ----
DATA_DIR     := "data"
IN_DIR       := DATA_DIR + "/in"
OUT_DIR      := DATA_DIR + "/out"
JOB_DIR      := DATA_DIR + "/jobs"
LOG_DIR      := DATA_DIR + "/logs"

# ---- Compute host absolute mount path for Docker Desktop on Windows ----
# Example: C:\Users\xxx\projects\subgen -> C:/Users/xxx/projects/subgen
PROJECT_ROOT_SLASH := replace(justfile_directory(), "\\", "/")
DATA_MOUNT         := PROJECT_ROOT_SLASH + "/" + DATA_DIR

# ---- Container paths ----
TEST_VIDEO   := "universe.webm"
VIDEO_PATH   := "/data/in/" + TEST_VIDEO
OUT_PATH     := "/data/out"

# Docker Desktop supports host.docker.internal
REDIS_URL    := "redis://host.docker.internal:6379/0"

RQ_QUEUE     := "subgen"
JOB_TIMEOUT  := "3600"
REQ_ID       := "pr6-gen-001"

# ---- IMPORTANT: disable MSYS path conversion for docker CLI args ----
DOCKER_ENV   := "MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL=*"

help:
	@echo "PR6 acceptance targets:"
	@echo "  just prep         # create data dirs"
	@echo "  just up           # start redis + api + worker"
	@echo "  just down         # stop & remove containers"
	@echo "  just logs-api     # follow api logs"
	@echo "  just logs-worker  # follow worker logs"
	@echo "  just smoke        # hit /healthz /readyz /metrics"
	@echo "  just gen          # POST /v1/subtitles/generate (prints response + HTTP code)"
	@echo "  just gen-save     # same as gen but also writes .tmp/resp.json"
	@echo "  just jobid        # prints job_id extracted from .tmp/resp.json"
	@echo "  just watch        # poll /v1/jobs/<job_id> until terminal state"
	@echo "  just result       # GET /v1/jobs/<job_id>/result"
	@echo "  just jobfiles     # list spec/status/result/debug.log on host"
	@echo "  just metrics      # fetch /metrics"
	@echo "  just fail-path    # illegal path test"
	@echo "  just fail-missing # missing input test"
	@echo
	@echo "Override vars: just IMAGE=subgen:pr6 PORT=8001 up"

prep:
	@echo
	@echo "==== Prep data dirs ===="
	mkdir -p "{{IN_DIR}}" "{{OUT_DIR}}" "{{JOB_DIR}}" "{{LOG_DIR}}" ".tmp"
	@echo "Project root: {{justfile_directory()}}"
	@echo "Host DATA_DIR: {{justfile_directory()}}/{{DATA_DIR}}"
	@echo "Docker mount DATA_MOUNT: {{DATA_MOUNT}}"
	@echo "Put your test video at: {{justfile_directory()}}/{{IN_DIR}}/{{TEST_VIDEO}}"

up: redis api worker
	@echo
	@echo "==== Up done ===="
	@{{DOCKER_ENV}} docker ps --filter "name=^/{{REDIS_NAME}}$$"  --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true
	@{{DOCKER_ENV}} docker ps --filter "name=^/{{API_NAME}}$$"    --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true
	@{{DOCKER_ENV}} docker ps --filter "name=^/{{WORKER_NAME}}$$" --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true

down:
	@echo
	@echo "==== Down containers ===="
	-{{DOCKER_ENV}} docker rm -f "{{WORKER_NAME}}" >/dev/null 2>&1 || true
	-{{DOCKER_ENV}} docker rm -f "{{API_NAME}}" >/dev/null 2>&1 || true
	-{{DOCKER_ENV}} docker rm -f "{{REDIS_NAME}}" >/dev/null 2>&1 || true
	@echo "OK"

redis:
	@echo
	@echo "==== Start Redis ===="
	-{{DOCKER_ENV}} docker rm -f "{{REDIS_NAME}}" >/dev/null 2>&1 || true
	{{DOCKER_ENV}} docker run -d --name "{{REDIS_NAME}}" -p 6379:6379 redis:7-alpine >/dev/null
	@echo "Redis started: {{REDIS_NAME}}"
	@{{DOCKER_ENV}} docker logs "{{REDIS_NAME}}" --tail 10 || true

api:
	@echo
	@echo "==== Start API ===="
	-{{DOCKER_ENV}} docker rm -f "{{API_NAME}}" >/dev/null 2>&1 || true
	@echo "DATA_MOUNT={{DATA_MOUNT}}"

	{{DOCKER_ENV}} docker run -d --name "{{API_NAME}}" \
	-p {{PORT}}:8000 \
	-v "{{DATA_MOUNT}}:/data" \
	-e SUBGEN_ALLOWED_ROOTS=/data \
	-e SUBGEN_DATA_ROOT=/data \
	-e SUBGEN_JOB_ROOT=/data/jobs \
	-e SUBGEN_ALLOW_CREATE_OUTPUT_DIR=true \
	-e SUBGEN_REDIS_URL="{{REDIS_URL}}" \
	-e SUBGEN_RQ_QUEUE_NAME="{{RQ_QUEUE}}" \
	-e SUBGEN_RQ_JOB_TIMEOUT_SEC="{{JOB_TIMEOUT}}" \
	-e SUBGEN_LOG_PATH=/data/logs/api.log \
	"{{IMAGE}}"

	@echo "API started: {{API_NAME}}  ->  {{BASE_URL}}"
	@{{DOCKER_ENV}} docker ps --filter "name=^/{{API_NAME}}$$" --format "table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}"

worker:
	@echo
	@echo "==== Start Worker ===="
	-{{DOCKER_ENV}} docker rm -f "{{WORKER_NAME}}" >/dev/null 2>&1 || true
	@echo "DATA_MOUNT={{DATA_MOUNT}}"

	{{DOCKER_ENV}} docker run -d --name "{{WORKER_NAME}}" \
	-v "{{DATA_MOUNT}}:/data" \
	-e SUBGEN_ALLOWED_ROOTS=/data \
	-e SUBGEN_DATA_ROOT=/data \
	-e SUBGEN_JOB_ROOT=/data/jobs \
	-e SUBGEN_ALLOW_CREATE_OUTPUT_DIR=true \
	-e SUBGEN_REDIS_URL="{{REDIS_URL}}" \
	-e SUBGEN_RQ_QUEUE_NAME="{{RQ_QUEUE}}" \
	-e SUBGEN_RQ_JOB_TIMEOUT_SEC="{{JOB_TIMEOUT}}" \
	"{{IMAGE}}" \
	python -m subgen.service.rq.worker_main

	@echo "Worker started: {{WORKER_NAME}}"
	@{{DOCKER_ENV}} docker ps --filter "name=^/{{WORKER_NAME}}$$" --format "table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}"

logs-api:
	@echo
	@echo "==== API logs ===="
	{{DOCKER_ENV}} docker logs -f "{{API_NAME}}"

logs-worker:
	@echo
	@echo "==== Worker logs ===="
	{{DOCKER_ENV}} docker logs -f "{{WORKER_NAME}}"

smoke:
	@echo
	@echo "==== Smoke endpoints ===="
	curl -fsS "{{BASE_URL}}/healthz" ; echo
	curl -fsS "{{BASE_URL}}/readyz"  ; echo
	curl -fsS "{{BASE_URL}}/metrics" | head -n 20 ; echo

gen:
	@echo
	@echo "==== POST /v1/subtitles/generate ===="
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: {{REQ_ID}}" \
	-d '{"video_path":"{{VIDEO_PATH}}","out_dir":"{{OUT_PATH}}"}' \
	-w '\nHTTP=%{http_code}\n'
	@echo

gen-save:
	@echo
	@echo "==== POST /v1/subtitles/generate -> .tmp/resp.json ===="
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: {{REQ_ID}}" \
	-d '{"video_path":"{{VIDEO_PATH}}","out_dir":"{{OUT_PATH}}"}' \
	| tee .tmp/resp.json
	@echo

jobid:
	@echo
	@echo "==== Extract job_id ===="
	test -f .tmp/resp.json || (echo "Missing .tmp/resp.json. Run: just gen-save"; exit 1)
	python -c 'import json; d=json.load(open(".tmp/resp.json","r",encoding="utf-8")); print(d.get("job_id") or d.get("id") or "")'

watch:
	@echo
	@echo "==== Poll /v1/jobs/<job_id> ===="
	@test -f .tmp/resp.json || (echo "Missing .tmp/resp.json. Run: just gen-save"; exit 1)

	@JOB_ID="$(python -c 'import json; d=json.load(open(".tmp/resp.json","r",encoding="utf-8")); print(d.get("job_id") or d.get("id") or "")')" ; \
	test -n "$JOB_ID" || (echo "job_id not found in .tmp/resp.json"; exit 1) ; \
	echo "job_id=$JOB_ID" ; \
	for i in $(seq 1 120); do \
	resp="$(curl -fsS "{{BASE_URL}}/v1/jobs/$JOB_ID")" ; \
	echo "$resp" ; \
	state="$(echo "$resp" | python -c 'import json,sys; data=json.load(sys.stdin); print(data.get("state") or data.get("status") or "")')" ; \
	if [[ "$state" == "succeeded" || "$state" == "failed" ]]; then \
	echo "terminal state=$state" ; \
	echo "$JOB_ID" > .tmp/job_id.txt ; \
	exit 0 ; \
	fi ; \
	sleep 2 ; \
	done ; \
	echo "Timeout waiting job" ; \
	exit 1

result:
	@echo
	@echo "==== GET /v1/jobs/<job_id>/result ===="
	test -f .tmp/job_id.txt || (echo "Missing .tmp/job_id.txt. Run: just watch"; exit 1)
	JOB_ID="$$(cat .tmp/job_id.txt)"
	curl -fsS "{{BASE_URL}}/v1/jobs/$$JOB_ID/result" ; echo

jobfiles:
	@echo
	@echo "==== Show job files on host ===="
	test -f .tmp/job_id.txt || (echo "Missing .tmp/job_id.txt. Run: just watch"; exit 1)
	JOB_ID="$$(cat .tmp/job_id.txt)"
	echo "Host path: {{JOB_DIR}}/$$JOB_ID"
	ls -lah "{{JOB_DIR}}/$$JOB_ID" || true
	echo "---- spec.json ----"
	test -f "{{JOB_DIR}}/$$JOB_ID/spec.json" && sed -n '1,200p' "{{JOB_DIR}}/$$JOB_ID/spec.json" || true
	echo "---- status.json ---"
	test -f "{{JOB_DIR}}/$$JOB_ID/status.json" && sed -n '1,200p' "{{JOB_DIR}}/$$JOB_ID/status.json" || true
	echo "---- result.json ---"
	test -f "{{JOB_DIR}}/$$JOB_ID/result.json" && sed -n '1,200p' "{{JOB_DIR}}/$$JOB_ID/result.json" || true
	echo "---- debug.log ----"
	test -f "{{JOB_DIR}}/$$JOB_ID/debug.log" && tail -n 200 "{{JOB_DIR}}/$$JOB_ID/debug.log" || true

metrics:
	@echo
	@echo "==== GET /metrics ===="
	curl -fsS "{{BASE_URL}}/metrics" | head -n 80 ; echo

fail-path:
	@echo
	@echo "==== Illegal path should be rejected ===="
	curl -i -s -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-d '{"video_path":"/etc/passwd","out_dir":"{{OUT_PATH}}"}' ; echo

fail-missing:
	@echo
	@echo "==== Missing input should fail ===="
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: pr6-miss-001" \
	-d '{"video_path":"/data/in/NOT_EXISTS.webm","out_dir":"{{OUT_PATH}}"}' \
	| tee .tmp/resp.json
	@echo
	@echo "Now: just watch && just result && just jobfiles"