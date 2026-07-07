"""Shared SSH helpers for UCL machine tools."""

from __future__ import annotations

import subprocess
from typing import Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]


def build_master_check_argv(master_host: str = "knuckles") -> list[str]:
    return ["ssh", "-O", "check", master_host]


def build_master_start_argv(master_host: str = "knuckles") -> list[str]:
    return ["ssh", "-MNf", master_host]


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
