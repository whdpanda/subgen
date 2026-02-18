from __future__ import annotations

import os
import secrets


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def build_worker_name() -> str:
    """
    Build a stable-ish worker name per process:
    - prefer pod name from POD_NAME/HOSTNAME/COMPUTERNAME
    - suffix with short POD_UID when available
    - otherwise use a random short suffix to avoid fixed worker names
    """
    base_name = _first_non_empty(
        os.getenv("POD_NAME"),
        os.getenv("HOSTNAME"),
        os.getenv("COMPUTERNAME"),
        "subgen-worker",
    )
    assert base_name is not None

    pod_uid = _first_non_empty(os.getenv("POD_UID"), os.getenv("K8S_POD_UID"))
    if pod_uid:
        suffix = pod_uid.replace("-", "")[:8]
    else:
        suffix = secrets.token_hex(4)

    return f"{base_name}-{suffix}"
