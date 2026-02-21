from __future__ import annotations

import os
import secrets
import time


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def build_worker_name() -> str:
    """
    Build a unique worker name per process startup:
    - prefer pod name from POD_NAME/HOSTNAME/COMPUTERNAME
    - include pod namespace/uid details when available
    - include process startup timestamp
    - always append random entropy to avoid any collision on fast restarts
    """
    base_name = _first_non_empty(
        os.getenv("POD_NAME"),
        os.getenv("HOSTNAME"),
        os.getenv("COMPUTERNAME"),
        "subgen-worker",
    )
    assert base_name is not None

    namespace = _first_non_empty(os.getenv("POD_NAMESPACE"), os.getenv("K8S_NAMESPACE"))
    pod_uid = _first_non_empty(os.getenv("POD_UID"), os.getenv("K8S_POD_UID"))
    startup_ts = format(time.time_ns(), "x")
    random_suffix = secrets.token_hex(4)

    parts = [base_name]
    if namespace:
        parts.append(namespace)
    parts.append(startup_ts)

    if pod_uid:
        parts.append(pod_uid.replace("-", "")[:8])
    parts.append(random_suffix)

    return "-".join(parts)
