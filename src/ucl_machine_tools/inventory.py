"""Read-only remote GPU and /tmp scratch inventory helpers."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
from collections import Counter
from typing import Any, Callable, Iterable

from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.ssh import build_remote_python_argv, describe_ssh_failure


INVENTORY_SENTINEL = "UCL_INVENTORY_JSON"
INVENTORY_SENTINEL_BEGIN = "UCL_INVENTORY_JSON_BEGIN"
INVENTORY_SENTINEL_END = "UCL_INVENTORY_JSON_END"
SCHEMA_VERSION = 1
LAB_PC_RESTART_TEXT = "Mon/Thu 19:30-midnight; may reboot anytime"
TIMESHARE_RESTART_TEXT = "no regular lab-PC window listed by TSG"

Runner = Callable[..., subprocess.CompletedProcess[str]]


def restart_text(policy: str) -> str:
    if policy == "lab_pc":
        return LAB_PC_RESTART_TEXT
    if policy == "timeshare":
        return TIMESHARE_RESTART_TEXT
    return "unknown"


def build_ssh_argv(host: HostSpec, *, timeout_seconds: int = 8) -> list[str]:
    return build_remote_python_argv(host.ssh_host, timeout_seconds=timeout_seconds)


def build_remote_probe_source(*, host: HostSpec, root: str, sizes: bool = False) -> str:
    if not root.startswith("/"):
        raise ValueError(f"root must be absolute: {root!r}")
    params = {
        "schema_version": SCHEMA_VERSION,
        "requested_host": host.name,
        "scratch_root": root,
        "restart_policy": host.restart_policy,
        "restart_text": restart_text(host.restart_policy),
        "sizes": bool(sizes),
    }
    params_json = json.dumps(params, sort_keys=True)
    return f"""#!/usr/bin/env python3
import csv
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

PARAMS = json.loads({params_json!r})
BEGIN = {json.dumps(INVENTORY_SENTINEL_BEGIN)}
END = {json.dumps(INVENTORY_SENTINEL_END)}


def run(cmd, timeout=5):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return None, repr(exc)


def parse_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_gpu_rows(text):
    rows = []
    for row in csv.reader(io.StringIO(text or "")):
        if not row or len(row) < 7:
            continue
        rows.append({{
            "uuid": row[0].strip(),
            "index": parse_int(row[1]),
            "name": row[2].strip(),
            "memory_total_mb": parse_int(row[3]),
            "memory_used_mb": parse_int(row[4]),
            "memory_free_mb": parse_int(row[5]),
            "utilization_gpu_percent": parse_int(row[6]),
            "processes": [],
        }})
    return rows


def parse_process_rows(text):
    rows = []
    for row in csv.reader(io.StringIO(text or "")):
        if not row or len(row) < 4:
            continue
        pid = parse_int(row[1])
        user = None
        if pid is not None:
            ps = run(["ps", "-o", "user=", "-p", str(pid)], timeout=2)
            if not isinstance(ps, tuple) and ps.returncode == 0:
                user = ps.stdout.strip() or None
        rows.append({{
            "gpu_uuid": row[0].strip(),
            "pid": pid,
            "user": user,
            "command": row[2].strip(),
            "used_memory_mb": parse_int(row[3]),
        }})
    return rows


def stat_path(path):
    p = Path(path)
    info = {{"path": str(p), "exists": p.exists()}}
    if not p.exists():
        return info
    try:
        st = os.statvfs(str(p))
        total = st.f_blocks * st.f_frsize
        available = st.f_bavail * st.f_frsize
        used = max(0, total - available)
        info.update({{
            "total_gb": total / (1024 ** 3),
            "available_gb": available / (1024 ** 3),
            "used_percent": (used / total * 100.0) if total else None,
        }})
    except Exception as exc:
        info["error"] = repr(exc)
    return info


def du_kib(path):
    proc = run(["du", "-sk", path], timeout=10)
    if isinstance(proc, tuple):
        return {{"path": path, "ok": False, "error": proc[1]}}
    if proc.returncode != 0:
        return {{"path": path, "ok": False, "error": (proc.stderr or proc.stdout).strip()}}
    first = proc.stdout.strip().split()
    return {{"path": path, "ok": True, "size_kib": parse_int(first[0]) if first else None}}


def main():
    root = PARAMS["scratch_root"]
    payload = {{
        "schema_version": PARAMS["schema_version"],
        "host": PARAMS["requested_host"],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_unix": time.time(),
        "ok": True,
        "gpus": [],
        "filesystems": [stat_path("/tmp"), stat_path(root)],
        "scratch": {{"root": root, "exists": Path(root).exists()}},
        "restart": {{
            "policy": PARAMS["restart_policy"],
            "text": PARAMS["restart_text"],
        }},
        "errors": [],
    }}

    gpu_proc = run([
        "nvidia-smi",
        "--query-gpu=uuid,index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], timeout=5)
    if isinstance(gpu_proc, tuple):
        payload["errors"].append("nvidia-smi gpu query failed: " + gpu_proc[1])
    elif gpu_proc.returncode != 0:
        payload["errors"].append("nvidia-smi gpu query failed: " + (gpu_proc.stderr or gpu_proc.stdout).strip())
    else:
        payload["gpus"] = parse_gpu_rows(gpu_proc.stdout)

    proc = run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], timeout=5)
    processes = []
    if isinstance(proc, tuple):
        payload["errors"].append("nvidia-smi process query failed: " + proc[1])
    elif proc.returncode == 0:
        processes = parse_process_rows(proc.stdout)
    elif proc.stderr.strip():
        payload["errors"].append("nvidia-smi process query failed: " + proc.stderr.strip())

    by_uuid = {{gpu.get("uuid"): gpu for gpu in payload["gpus"]}}
    for item in processes:
        gpu = by_uuid.get(item.get("gpu_uuid"))
        if gpu is not None:
            gpu.setdefault("processes", []).append(item)

    if PARAMS["sizes"]:
        payload["sizes"] = [du_kib(root)]

    print(BEGIN)
    print(json.dumps(payload, sort_keys=True))
    print(END)


try:
    main()
except Exception as exc:
    payload = {{
        "schema_version": PARAMS["schema_version"],
        "host": PARAMS["requested_host"],
        "hostname": socket.gethostname() if "socket" in globals() else None,
        "ok": False,
        "gpus": [],
        "filesystems": [],
        "scratch": {{"root": PARAMS["scratch_root"], "exists": False}},
        "restart": {{"policy": PARAMS["restart_policy"], "text": PARAMS["restart_text"]}},
        "errors": ["remote_exception: " + repr(exc)],
    }}
    print(BEGIN)
    print(json.dumps(payload, sort_keys=True))
    print(END)
    sys.exit(0)
"""


def parse_sentinel_stdout(stdout: str) -> dict[str, Any]:
    block_payloads: list[str] = []
    lines = stdout.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line == INVENTORY_SENTINEL_BEGIN:
            end_idx = None
            for candidate in range(idx + 1, len(lines)):
                if lines[candidate].strip() == INVENTORY_SENTINEL_END:
                    end_idx = candidate
                    break
            if end_idx is None:
                raise ValueError("inventory sentinel begin found without end sentinel")
            block_payloads.append("\n".join(lines[idx + 1 : end_idx]).strip())
            idx = end_idx
        elif line.startswith(INVENTORY_SENTINEL + " "):
            block_payloads.append(line.split(" ", 1)[1].strip())
        idx += 1

    if not block_payloads:
        raise ValueError("inventory sentinel not found")
    if len(block_payloads) > 1:
        raise ValueError("multiple inventory sentinels found")
    try:
        payload = json.loads(block_payloads[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed inventory sentinel JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("inventory sentinel JSON must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported inventory schema_version: {payload.get('schema_version')!r}")
    for key in ("host", "ok", "gpus", "filesystems", "scratch", "restart", "errors"):
        if key not in payload:
            raise ValueError(f"inventory payload missing required key: {key}")
    return payload


def _tmp_available_gb(payload: dict[str, Any]) -> float | None:
    for fs in payload.get("filesystems", []) or []:
        if fs.get("path") == "/tmp":
            value = fs.get("available_gb")
            return float(value) if value is not None else None
    return None


def classify(
    payload: dict[str, Any],
    *,
    min_tmp_free_gb: float = 50.0,
    min_free_vram_gb: float = 4.0,
    max_gpu_util_percent: float = 20.0,
) -> str:
    if not payload.get("ok", False):
        return "unreachable"
    tmp_free = _tmp_available_gb(payload)
    if tmp_free is not None and tmp_free < min_tmp_free_gb:
        return "storage-low"
    gpus = payload.get("gpus", []) or []
    if not gpus:
        return "no-gpu"
    for gpu in gpus:
        processes = gpu.get("processes", []) or []
        free_mb = gpu.get("memory_free_mb")
        if free_mb is None:
            total = gpu.get("memory_total_mb")
            used = gpu.get("memory_used_mb")
            free_mb = (total - used) if total is not None and used is not None else None
        util = gpu.get("utilization_gpu_percent")
        if processes:
            continue
        if free_mb is not None and free_mb < min_free_vram_gb * 1024:
            continue
        if util is not None and util > max_gpu_util_percent:
            continue
        return "ready"
    return "busy"


def collect_one(
    host: HostSpec,
    *,
    runner: Runner = subprocess.run,
    timeout_seconds: int = 8,
    root: str | None = None,
    sizes: bool = False,
    debug: bool = False,
    min_tmp_free_gb: float = 50.0,
    min_free_vram_gb: float = 4.0,
) -> dict[str, Any]:
    argv = build_ssh_argv(host, timeout_seconds=timeout_seconds)
    probe = build_remote_probe_source(host=host, root=root or host.scratch_root, sizes=sizes)
    try:
        proc = runner(
            argv,
            input=probe,
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 3,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _error_row(host, "ssh-timeout", f"ssh timed out after {exc.timeout}s")
    except Exception as exc:  # noqa: BLE001 - diagnostic wrapper around user SSH config.
        return _error_row(host, "ssh-failed", repr(exc))

    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    returncode = int(getattr(proc, "returncode", 1))
    try:
        payload = parse_sentinel_stdout(stdout)
    except ValueError as exc:
        if returncode != 0:
            status = "unreachable" if returncode == 255 else "ssh-failed"
            message = describe_ssh_failure(returncode, stdout=stdout, stderr=stderr)
        else:
            status = "no-sentinel" if "sentinel not found" in str(exc) else "parse-error"
            message = str(exc)
        row = _error_row(host, status, message)
        row["ssh_returncode"] = returncode
        if debug and stderr:
            row["stderr_tail"] = stderr[-500:]
        if debug and stdout:
            row["stdout_tail"] = stdout[-500:]
        return row

    payload["ssh_host"] = host.ssh_host
    payload["expected_gpu_count"] = host.expected_gpu_count
    payload["expected_gpu_name"] = host.expected_gpu_name
    payload["restart"] = payload.get("restart") or {"policy": host.restart_policy, "text": restart_text(host.restart_policy)}
    if returncode != 0:
        payload.setdefault("errors", []).append(f"ssh exited {returncode}")
    payload["status"] = classify(
        payload,
        min_tmp_free_gb=min_tmp_free_gb,
        min_free_vram_gb=min_free_vram_gb,
    )
    return payload


def _error_row(host: HostSpec, status: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "host": host.name,
        "ssh_host": host.ssh_host,
        "ok": False,
        "status": status,
        "gpus": [],
        "filesystems": [],
        "scratch": {"root": host.scratch_root, "exists": False},
        "restart": {"policy": host.restart_policy, "text": restart_text(host.restart_policy)},
        "errors": [message],
    }


def collect(
    hosts: Iterable[HostSpec],
    *,
    runner: Runner = subprocess.run,
    jobs: int = 4,
    timeout_seconds: int = 8,
    root: str | None = None,
    sizes: bool = False,
    debug: bool = False,
    min_tmp_free_gb: float = 50.0,
    min_free_vram_gb: float = 4.0,
) -> list[dict[str, Any]]:
    host_list = list(hosts)
    if not host_list:
        raise ValueError("at least one host is required")
    if jobs <= 0:
        raise ValueError("jobs must be positive")

    rows: list[dict[str, Any] | None] = [None] * len(host_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(host_list))) as executor:
        futures = {
            executor.submit(
                collect_one,
                host,
                runner=runner,
                timeout_seconds=timeout_seconds,
                root=root,
                sizes=sizes,
                debug=debug,
                min_tmp_free_gb=min_tmp_free_gb,
                min_free_vram_gb=min_free_vram_gb,
            ): idx
            for idx, host in enumerate(host_list)
        }
        for future in concurrent.futures.as_completed(futures):
            rows[futures[future]] = future.result()
    return [row for row in rows if row is not None]


def _fmt_gb(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value >= 100:
        return f"{value:.0f}G"
    if value >= 10:
        return f"{value:.1f}G"
    return f"{value:.2f}G"


def _compact_gpu_name(name: str | None) -> str:
    if not name:
        return "n/a"
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _best_gpu(row: dict[str, Any]) -> dict[str, Any] | None:
    gpus = row.get("gpus", []) or []
    if not gpus:
        return None
    return max(gpus, key=lambda gpu: float(gpu.get("memory_free_mb") or 0))


def _gpu_summary(row: dict[str, Any]) -> str:
    gpus = row.get("gpus", []) or []
    if not gpus:
        return "0/0"
    free = sum(1 for gpu in gpus if not (gpu.get("processes", []) or []))
    return f"{free}/{len(gpus)}"


def _scratch_summary(row: dict[str, Any]) -> str:
    scratch = row.get("scratch") or {}
    return "yes" if scratch.get("exists") else "no"


def _tmp_summary(row: dict[str, Any]) -> str:
    return _fmt_gb(_tmp_available_gb(row))


def _note(row: dict[str, Any]) -> str:
    errors = row.get("errors", []) or []
    if errors:
        text = str(errors[0]).replace("\n", " ")
        return text[:48] + ("..." if len(text) > 48 else "")
    gpu = _best_gpu(row)
    if gpu and (gpu.get("processes", []) or []):
        users = sorted({str(proc.get("user") or proc.get("pid") or "?") for proc in gpu.get("processes", [])})
        return "users:" + ",".join(users[:3])
    return "-"


def format_table(rows: list[dict[str, Any]]) -> str:
    table_rows: list[list[str]] = []
    for row in rows:
        best = _best_gpu(row)
        best_gpu = "n/a"
        vram = "n/a"
        if best is not None:
            best_gpu = f"{best.get('index', '?')} {_compact_gpu_name(best.get('name'))}"
            free_mb = best.get("memory_free_mb")
            if free_mb is None and best.get("memory_total_mb") is not None and best.get("memory_used_mb") is not None:
                free_mb = best["memory_total_mb"] - best["memory_used_mb"]
            vram = _fmt_gb((float(free_mb) / 1024) if free_mb is not None else None)
        table_rows.append(
            [
                str(row.get("host", "n/a")),
                str(row.get("status", "unknown")),
                _gpu_summary(row),
                best_gpu,
                vram,
                _tmp_summary(row),
                _scratch_summary(row),
                str((row.get("restart") or {}).get("text") or "unknown"),
                str(row.get("ssh_host") or row.get("host") or "n/a"),
                _note(row),
            ]
        )
    headers = ["host", "status", "gpu", "best_gpu", "vram", "tmp_free", "tmp_scratch", "restart", "ssh", "note"]
    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines = ["  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    for row in table_rows:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def to_jsonable(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("status", "unknown")) for row in rows)
    summary = {key: counts[key] for key in sorted(counts)}
    summary["total"] = len(rows)
    return {"schema_version": SCHEMA_VERSION, "summary": summary, "hosts": rows}
