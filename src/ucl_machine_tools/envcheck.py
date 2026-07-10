"""Remote scratch and environment preflight checks."""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable

from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.ssh import build_remote_python_argv


Runner = Callable[..., subprocess.CompletedProcess]
ENV_BEGIN = "UCL_ENV_JSON_BEGIN"
ENV_END = "UCL_ENV_JSON_END"


def env_source(*, remote_root: str, create: bool, gpu: str | None) -> str:
    return f"""
import json
import os
import shutil
import subprocess
from pathlib import Path
BEGIN={json.dumps(ENV_BEGIN)}
END={json.dumps(ENV_END)}
ROOT=Path({json.dumps(remote_root)})
CREATE={bool(create)!r}
GPU={gpu!r}
errors = []
created = False
if CREATE:
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        created = True
    except Exception as exc:
        errors.append(f"create root failed: {{type(exc).__name__}}: {{exc}}")
tmp_usage = shutil.disk_usage("/tmp")
cuda_visibility_candidates = [
    Path("/usr/local/cuda/CUDA_VISIBILITY.csh"),
    Path("/opt/cuda/scripts/CUDA_VISIBILITY.csh"),
]
cuda_visibility = next((path for path in cuda_visibility_candidates if path.exists()), cuda_visibility_candidates[0])
python_setup = Path("/opt/Python/Python-3.11.5_Setup.csh")
gpu_info = None
if GPU is not None:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    gpu_info = {{"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}}
    if proc.returncode != 0:
        errors.append("nvidia-smi failed")
payload = {{
    "schema_version": 1,
    "remote_root": str(ROOT),
    "root_exists": ROOT.exists(),
    "root_created": created,
    "tmp_free_gb": round(tmp_usage.free / (1024 ** 3), 2),
    "cuda_visibility_script": str(cuda_visibility),
    "cuda_visibility_candidates": [str(path) for path in cuda_visibility_candidates],
    "cuda_visibility_exists": cuda_visibility.exists(),
    "python_setup_script": str(python_setup),
    "python_setup_exists": python_setup.exists(),
    "gpu": GPU,
    "gpu_info": gpu_info,
    "ok": not errors,
    "errors": errors,
}}
print(BEGIN)
print(json.dumps(payload, sort_keys=True))
print(END)
"""


def parse_env_output(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    payloads = []
    for idx, line in enumerate(lines):
        if line.strip() != ENV_BEGIN:
            continue
        for end_idx in range(idx + 1, len(lines)):
            if lines[end_idx].strip() == ENV_END:
                payloads.append("\n".join(lines[idx + 1 : end_idx]).strip())
                break
    if len(payloads) != 1:
        raise RuntimeError("env sentinel not found" if not payloads else "multiple env sentinels found")
    payload = json.loads(payloads[0])
    if payload.get("schema_version") != 1:
        raise RuntimeError("invalid env sentinel payload")
    return payload


def run_env_check(
    host: HostSpec,
    *,
    remote_root: str,
    create: bool,
    gpu: str | None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    proc = runner(
        build_remote_python_argv(host.ssh_host),
        input=env_source(remote_root=remote_root, create=create, gpu=gpu),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        detail = "\n".join(line for line in detail.splitlines() if "VBoxManage" not in line and "VirtualBox" not in line).strip()
        raise RuntimeError(detail or f"remote env check failed on {host.name}")
    return parse_env_output(getattr(proc, "stdout", "") or "")
