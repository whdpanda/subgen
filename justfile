# =========================
# PR6 Acceptance justfile
# =========================

set shell := ["C:/Program Files/Git/bin/bash.exe", "-eu", "-o", "pipefail", "-c"]

# ---- Config (override via: `just VAR=value target`) ----
HOST         := "127.0.0.1"
PORT         := "8000"
BASE_URL     := "http://" + HOST + ":" + PORT

REDIS_NAME   := "subgen-redis"
API_NAME     := "subgen-api"
WORKER_NAME  := "subgen-worker"

# ---- GPU toggle ----
# Use: `just USE_GPU=true ...` (default true) or `just USE_GPU=false ...`
USE_GPU := "true"
GPU_ARGS := if USE_GPU == "true" { "--gpus all" } else { "" }

# ---- Build args defaults ----
# Extras: agent / rag / agent_rag / all
INSTALL_EXTRAS := "agent_rag"
TORCH_VARIANT  := if USE_GPU == "true" { "cu121" } else { "cpu" }

# ---- Image tag ----
IMAGE := if USE_GPU == "true" { "subgen:pr6-gpu" } else { "subgen:pr6-cpu" }

# ---- Host paths (relative; created under repo) ----
DATA_DIR     := "data"
IN_DIR       := DATA_DIR + "/in"
OUT_DIR      := DATA_DIR + "/out"
JOB_DIR      := DATA_DIR + "/jobs"
LOG_DIR      := DATA_DIR + "/logs"

# ---- Compute host absolute mount path for Docker Desktop on Windows ----
PROJECT_ROOT_SLASH := replace(justfile_directory(), "\\", "/")
DATA_MOUNT         := PROJECT_ROOT_SLASH + "/" + DATA_DIR

# ---- Container paths ----
TEST_VIDEO   := "universe.webm"
VIDEO_PATH   := "/data/in/" + TEST_VIDEO
OUT_PATH     := "/data/out"

# ---- Burn config (host-relative) ----
# You can override: just BURN_SRT=foo.srt burn-save
BURN_SRT      := "universe.zh.fixed.srt"
BURN_OUT_NAME := "universe.burned.mp4"
BURN_OUT_PATH := "/data/out/" + BURN_OUT_NAME

# Docker Desktop supports host.docker.internal
REDIS_URL    := "redis://host.docker.internal:6379/0"

RQ_QUEUE     := "subgen"
JOB_TIMEOUT  := "3600"
REQ_ID       := "pr6-gen-001"

# ---- Watch settings ----
WATCH_TIMEOUT_SEC := JOB_TIMEOUT
WATCH_INTERVAL_SEC := "2"

# ---- IMPORTANT: disable MSYS path conversion for docker CLI args ----
DOCKER_ENV   := "MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL=*"

help:
	@echo "PR6 acceptance targets:"
	@echo "  just prep           # create data dirs"
	@echo "  just build          # docker build (uses USE_GPU/INSTALL_EXTRAS)"
	@echo "  just rebuild        # docker build --no-cache"
	@echo "  just verify-torch   # check torch/cuda in built image"
	@echo "  just up             # start redis + api + worker"
	@echo "  just down           # stop & remove containers"
	@echo "  just logs-api       # follow api logs"
	@echo "  just logs-worker    # follow worker logs"
	@echo "  just smoke          # hit /healthz /readyz /metrics"
	@echo "  just gen            # POST /v1/subtitles/generate"
	@echo "  just gen-save       # POST -> .tmp/resp.json"
	@echo "  just jobid          # extract job_id from .tmp/resp.json"
	@echo "  just watch          # poll /v1/jobs/<job_id> until terminal"
	@echo "  just result         # GET /v1/jobs/<job_id>/result"
	@echo "  just jobfiles       # list spec/status/result/debug.log on host"
	@echo "  just metrics        # fetch /metrics"
	@echo "  just fail-path      # illegal path test"
	@echo "  just fail-missing   # missing input test"
	@echo
	@echo "Current build/runtime vars:"
	@echo "  USE_GPU={{USE_GPU}}  TORCH_VARIANT={{TORCH_VARIANT}}  GPU_ARGS='{{GPU_ARGS}}'"
	@echo "  INSTALL_EXTRAS={{INSTALL_EXTRAS}}"
	@echo "  IMAGE={{IMAGE}}"
	@echo
	@echo "Examples:"
	@echo "  just USE_GPU=true  build up"
	@echo "  just USE_GPU=false build up"
	@echo "  just INSTALL_EXTRAS=all USE_GPU=false rebuild up"

prep:
	@echo
	@echo "==== Prep data dirs ===="
	mkdir -p "{{IN_DIR}}" "{{OUT_DIR}}" "{{JOB_DIR}}" "{{LOG_DIR}}" ".tmp"
	@echo "Project root: {{justfile_directory()}}"
	@echo "Host DATA_DIR: {{justfile_directory()}}/{{DATA_DIR}}"
	@echo "Docker mount DATA_MOUNT: {{DATA_MOUNT}}"
	@echo "Put your test video at: {{justfile_directory()}}/{{IN_DIR}}/{{TEST_VIDEO}}"

build:
	@echo
	@echo "==== Docker build ===="
	@echo "IMAGE={{IMAGE}}  INSTALL_EXTRAS={{INSTALL_EXTRAS}}  TORCH_VARIANT={{TORCH_VARIANT}}"
	{{DOCKER_ENV}} docker build -t "{{IMAGE}}" \
	--build-arg INSTALL_EXTRAS="{{INSTALL_EXTRAS}}" \
	--build-arg TORCH_VARIANT="{{TORCH_VARIANT}}" \
	.

rebuild:
	@echo
	@echo "==== Docker build --no-cache ===="
	@echo "IMAGE={{IMAGE}}  INSTALL_EXTRAS={{INSTALL_EXTRAS}}  TORCH_VARIANT={{TORCH_VARIANT}}"
	{{DOCKER_ENV}} docker build --no-cache -t "{{IMAGE}}" \
	--build-arg INSTALL_EXTRAS="{{INSTALL_EXTRAS}}" \
	--build-arg TORCH_VARIANT="{{TORCH_VARIANT}}" \
	.

verify-torch:
	@echo
	@echo "==== Verify torch/cuda inside image ===="
	{{DOCKER_ENV}} docker run --rm {{GPU_ARGS}} "{{IMAGE}}" \
	python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('is_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"

up: redis api worker
	@echo
	@echo "==== Up done ===="
	@{{DOCKER_ENV}} docker ps --filter 'name=^/{{REDIS_NAME}}$$'  --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true
	@{{DOCKER_ENV}} docker ps --filter 'name=^/{{API_NAME}}$$'    --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true
	@{{DOCKER_ENV}} docker ps --filter 'name=^/{{WORKER_NAME}}$$' --format 'table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}' || true

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
	@echo "USE_GPU={{USE_GPU}}  GPU_ARGS={{GPU_ARGS}}"
	@echo "IMAGE={{IMAGE}}"

	# API 默认不需要 GPU；如果你的 API 里也会直接跑 pipeline，可把 {{GPU_ARGS}} 加进来。
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
	-e OPENAI_API_KEY \
	"{{IMAGE}}"

	@echo "API started: {{API_NAME}}  ->  {{BASE_URL}}"
	@{{DOCKER_ENV}} docker ps --filter 'name=^/{{API_NAME}}$$' --format "table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}"

worker:
	@echo
	@echo "==== Start Worker ===="
	-{{DOCKER_ENV}} docker rm -f "{{WORKER_NAME}}" >/dev/null 2>&1 || true
	@echo "DATA_MOUNT={{DATA_MOUNT}}"
	@echo "USE_GPU={{USE_GPU}}  GPU_ARGS={{GPU_ARGS}}"
	@echo "IMAGE={{IMAGE}}"

	{{DOCKER_ENV}} docker run -d --name "{{WORKER_NAME}}" \
	{{GPU_ARGS}} \
	-v "{{DATA_MOUNT}}:/data" \
	-e SUBGEN_ALLOWED_ROOTS=/data \
	-e SUBGEN_DATA_ROOT=/data \
	-e SUBGEN_JOB_ROOT=/data/jobs \
	-e SUBGEN_ALLOW_CREATE_OUTPUT_DIR=true \
	-e SUBGEN_REDIS_URL="{{REDIS_URL}}" \
	-e SUBGEN_RQ_QUEUE_NAME="{{RQ_QUEUE}}" \
	-e SUBGEN_RQ_JOB_TIMEOUT_SEC="{{JOB_TIMEOUT}}" \
	-e OPENAI_API_KEY \
	"{{IMAGE}}" \
	python -m subgen.service.rq.worker_main

	@echo "Worker started: {{WORKER_NAME}}"
	@{{DOCKER_ENV}} docker ps --filter 'name=^/{{WORKER_NAME}}$$' --format "table {{ "{{" }}.Names{{ "}}" }}\t{{ "{{" }}.Status{{ "}}" }}\t{{ "{{" }}.Ports{{ "}}" }}"

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
	@test -f .tmp/resp.json || (echo "Missing .tmp/resp.json. Run: just gen-save"; exit 1)
	@python -c 'import json; d=json.load(open(".tmp/resp.json","r",encoding="utf-8")); print(d.get("job_id") or d.get("id") or "")'

watch:
	#!C:/Progra~1/Git/usr/bin/bash.exe
	set -euo pipefail

	echo
	echo "==== Poll /v1/jobs/<job_id> ===="
	test -f .tmp/resp.json || { echo "Missing .tmp/resp.json. Run: just gen-save"; exit 1; }

	JOB_ID="$(python -c 'import json; d=json.load(open(".tmp/resp.json","r",encoding="utf-8")); print(d.get("job_id") or d.get("id") or "")')"
	test -n "$JOB_ID" || { echo "job_id not found in .tmp/resp.json"; exit 1; }

	echo "job_id=$JOB_ID"
	echo "$JOB_ID" > .tmp/job_id.txt

	max_iter=$(( ({{WATCH_TIMEOUT_SEC}} + {{WATCH_INTERVAL_SEC}} - 1) / {{WATCH_INTERVAL_SEC}} ))

	for i in $(seq 1 "$max_iter"); do
	resp="$(curl -fsS "{{BASE_URL}}/v1/jobs/$JOB_ID")"
	echo "$resp"

	state="$(echo "$resp" | python -c 'import json,sys; data=json.load(sys.stdin); print(data.get("state") or data.get("status") or "")')"

	if [[ "$state" == "succeeded" || "$state" == "failed" ]]; then
	echo "terminal state=$state"
	exit 0
	fi

	sleep "{{WATCH_INTERVAL_SEC}}"
	done

	echo "Timeout waiting job (state not terminal yet). You can rerun: just watch"
	exit 0

result:
	#!C:/Progra~1/Git/usr/bin/bash.exe
	set -euo pipefail

	echo
	echo "==== GET /v1/jobs/<job_id>/result ===="
	test -f .tmp/job_id.txt || { echo "Missing .tmp/job_id.txt. Run: just watch"; exit 1; }

	JOB_ID="$(cat .tmp/job_id.txt)"
	curl -fsS "{{BASE_URL}}/v1/jobs/$JOB_ID/result"
	echo

jobfiles:
	#!C:/Progra~1/Git/usr/bin/bash.exe
	set -euo pipefail

	echo
	echo "==== Show job files on host ===="
	test -f .tmp/job_id.txt || { echo "Missing .tmp/job_id.txt. Run: just watch"; exit 1; }

	JOB_ID="$(cat .tmp/job_id.txt)"
	echo "Host path: {{JOB_DIR}}/$JOB_ID"

	ls -lah "{{JOB_DIR}}/$JOB_ID" || true

	echo "---- spec.json ----"
	test -f "{{JOB_DIR}}/$JOB_ID/spec.json" && sed -n '1,200p' "{{JOB_DIR}}/$JOB_ID/spec.json" || true

	echo "---- status.json ----"
	test -f "{{JOB_DIR}}/$JOB_ID/status.json" && sed -n '1,200p' "{{JOB_DIR}}/$JOB_ID/status.json" || true

	echo "---- result.json ----"
	test -f "{{JOB_DIR}}/$JOB_ID/result.json" && sed -n '1,200p' "{{JOB_DIR}}/$JOB_ID/result.json" || true

	echo "---- debug.log ----"
	test -f "{{JOB_DIR}}/$JOB_ID/debug.log" && tail -n 200 "{{JOB_DIR}}/$JOB_ID/debug.log" || true

metrics:
	@echo
	@echo "==== /metrics (head) ===="
	curl -fsS "{{BASE_URL}}/metrics" | head -n 50 ; echo

fail-path:
	@echo
	@echo "==== Illegal path test ===="
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: pr6-illegal-001" \
	-d '{"video_path":"/etc/passwd","out_dir":"{{OUT_PATH}}"}' \
	| tee .tmp/resp.json
	@echo
	@echo "Now: just watch && just result && just jobfiles"

fail-missing:
	@echo
	@echo "==== Missing input test ===="
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/generate" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: pr6-miss-001" \
	-d '{"video_path":"/data/in/NOT_EXISTS.webm","out_dir":"{{OUT_PATH}}"}' \
	| tee .tmp/resp.json
	@echo
	@echo "Now: just watch && just result && just jobfiles"

burn-save:
	@echo
	@echo "==== POST /v1/subtitles/burn -> .tmp/resp.json ===="
	@echo "Using video: {{VIDEO_PATH}}"
	@echo "Using srt  : /data/out/{{BURN_SRT}}"
	@echo "Out video  : {{BURN_OUT_PATH}}"
	curl -fsS -X POST "{{BASE_URL}}/v1/subtitles/burn" \
	-H "Content-Type: application/json" \
	-H "X-Request-Id: pr6-burn-001" \
	-d '{"video_path":"{{VIDEO_PATH}}","srt_path":"/data/out/{{BURN_SRT}}","out_path":"{{BURN_OUT_PATH}}"}' \
	| tee .tmp/resp.json
	@echo
