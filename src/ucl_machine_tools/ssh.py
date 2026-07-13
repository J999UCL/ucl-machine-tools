"""Shared SSH helpers for UCL machine tools."""

from __future__ import annotations

import subprocess
from typing import Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]


def describe_ssh_failure(
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> str:
    """Return a concise transport diagnosis without exposing login noise."""
    combined = "\n".join(part for part in (stderr, stdout) if part)
    if "No route to host" in combined:
        return "target host is unreachable from the jump host"
    if "Connection refused" in combined:
        return "SSH connection was refused"
    if "Permission denied" in combined:
        return "SSH authentication failed"
    if "Could not resolve hostname" in combined or "Name or service not known" in combined:
        return "hostname could not be resolved"
    if "Stdio forwarding request failed" in combined or "UNKNOWN port 65535" in combined:
        return "jump-host forwarding failed; the target may be unreachable"
    if returncode == 255:
        return "target host or jump connection is unreachable"
    return f"SSH exited {returncode} before the remote probe started"


def build_master_check_argv(master_host: str = "knuckles") -> list[str]:
    return ["ssh", "-O", "check", master_host]


def build_master_start_argv(master_host: str = "knuckles") -> list[str]:
    return ["ssh", "-MNf", master_host]


def build_remote_python_argv(host: str, *, timeout_seconds: int | None = None) -> list[str]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
    if timeout_seconds is not None:
        argv += ["-o", f"ConnectTimeout={int(timeout_seconds)}"]
    return [*argv, host, "python3", "-"]


def ensure_knuckles_master(
    *,
    runner: Runner = subprocess.run,
    master_host: str = "knuckles",
) -> str:
    check = runner(
        build_master_check_argv(master_host),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(check, "returncode", 1)) == 0:
        return "existing"

    start = runner(
        build_master_start_argv(master_host),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(start, "returncode", 1)) != 0:
        stderr = (getattr(start, "stderr", "") or "").strip()
        stdout = (getattr(start, "stdout", "") or "").strip()
        detail = stderr or stdout or f"exit {getattr(start, 'returncode', 'unknown')}"
        raise RuntimeError(f"failed to start SSH master connection for {master_host}: {detail}")
    return "started"
