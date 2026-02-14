# src/subgen/api/main.py
from __future__ import annotations

import os

# IMPORTANT:
# Export the real application created in app.py (which includes api_router).
from subgen.api.app import app  # noqa: F401


def run() -> None:
    """
    Optional programmatic runner:
    python -m subgen.api.main
    """
    import uvicorn  # local import to keep import graph light

    host = os.getenv("SUBGEN_API_HOST", "0.0.0.0")
    port = int(os.getenv("SUBGEN_API_PORT", "8000"))

    # Keep this stable for Docker CMD and local running.
    uvicorn.run("subgen.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()