from __future__ import annotations

from fastapi import FastAPI, HTTPException

from subgen.core.pipeline import run_pipeline
from subgen.service.schemas import RunPipelineRequest, RunPipelineResponse

app = FastAPI(title="subgen API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/pipeline/run", response_model=RunPipelineResponse)
def run_pipeline_endpoint(req: RunPipelineRequest) -> RunPipelineResponse:
    try:
        result = run_pipeline(req.to_config())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc
    return RunPipelineResponse.from_result(result)
