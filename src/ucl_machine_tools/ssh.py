"""Shared SSH helpers for UCL machine tools."""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Sequence

from ucl_machine_tools import rsync_transport


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


def build_remote_argv(
    host: str,
    command: Sequence[str],
    *,
    timeout_seconds: int | None = None,
    forward_agent: bool = False,
    ssh_executable: str = "ssh",
) -> list[str]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    handshake_timeout = (
        rsync_transport.DEFAULT_HANDSHAKE_TIMEOUT_SECONDS
        if timeout_seconds is None
        else float(timeout_seconds)
    )
    return rsync_transport.build_transport_argv(
        host,
        command,
        ssh_executable=ssh_executable,
        handshake_timeout_seconds=handshake_timeout,
        forward_agent=forward_agent,
    )


def build_remote_python_argv(host: str, *, timeout_seconds: int | None = None) -> list[str]:
    return build_remote_argv(host, ("python3", "-"), timeout_seconds=timeout_seconds)


def ensure_ssh_agent(*, runner: Runner = subprocess.run) -> None:
    """Require a locally available agent with at least one loaded identity."""

    if not os.environ.get("SSH_AUTH_SOCK"):
        raise RuntimeError("SSH_AUTH_SOCK is not set; cannot forward the controller SSH agent")
    result = runner(
        ["ssh-add", "-l"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(result, "returncode", 1)) == 0:
        return
    detail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    if "no identities" in detail.lower():
        raise RuntimeError("the controller SSH agent has no loaded identities")
    raise RuntimeError(f"could not query the controller SSH agent: {detail or 'ssh-add exited unsuccessfully'}")


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
        detail = (stderr or stdout).strip() or f"exit {getattr(start, 'returncode', 'unknown')}"
        raise RuntimeError(f"failed to start SSH master connection for {master_host}: {detail}")
    return "started"
