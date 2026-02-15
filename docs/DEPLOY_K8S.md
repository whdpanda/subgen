# SubGen on Kubernetes

## 1. Prerequisites
- A working container image `subgen:latest` that contains:
  - API entry: `uvicorn subgen.api.app:app`
  - Worker entry: `python -m subgen.service.rq.worker_main`
- A cluster with a default StorageClass (for PVC).

## 2. Apply manifests
From repo root:

```bash
kubectl apply -k k8s/
```

## 3. Verify
```bash
kubectl -n subgen get pods
kubectl -n subgen get svc
```

Health:

```bash
kubectl -n subgen port-forward svc/subgen-api 8080:80
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/readyz
curl -s http://127.0.0.1:8080/metrics
```

## 4. Storage layout (inside /data)
- `/data/jobs/<job_id>/spec.json`
- `/data/jobs/<job_id>/status.json`
- `/data/jobs/<job_id>/result.json`
- `/data/jobs/<job_id>/debug.log`

## 5. Scaling guidance
- **API**: typically 1–2 replicas (stateless; ensure shared `/data` if you scale horizontally).
- **Worker**: current manifest defaults to **1 replica** because the shared PVC is `ReadWriteOnce`.
  - If you need multiple workers, use an RWX storage class (or guarantee same-node scheduling before increasing replicas).
- **Redis**: keep 1 replica for now. For production HA, use managed Redis or a Redis operator.

## 6. Security notes
- `SUBGEN_ALLOWED_ROOTS` limits file access. In this minimal setup it is `/data`.
- Mount only what you need under `/data`.
