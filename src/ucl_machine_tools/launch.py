"""Generic tar-over-SSH launcher for UCL machines."""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ucl_machine_tools.hosts import HostSpec


Runner = Callable[..., subprocess.CompletedProcess]
Popener = Callable[..., subprocess.Popen]

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REMOTE_ROOT = "/tmp/ucl-machine-tools/launchers"


@dataclass(frozen=True)
class LaunchPlan:
    host: HostSpec
    local_dir: Path
    script_rel: str
    remote_dir: str
    log_path: str
    generated_session: str
    requested_session: str | None
    window: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    replace: bool = False
    new_session: bool = False


@dataclass(frozen=True)
class TmuxDecision:
    mode: str
    session: str
    window: str
    existing_sessions: tuple[str, ...]


def utc_run_id(script_rel: str) -> str:
    stem = Path(script_rel).stem or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}_{timestamp}"


def validate_name(value: str, label: str) -> None:
    if not value or not _SAFE_NAME_RE.match(value):
        raise ValueError(f"{label} may only contain letters, numbers, dot, dash, and underscore: {value!r}")


def validate_remote_dir(remote_dir: str) -> None:
    if not remote_dir.startswith("/"):
        raise ValueError(f"remote_dir must be absolute: {remote_dir!r}")
    normalized = posixpath.normpath(remote_dir)
    if normalized == "/" or ".." in normalized.split("/"):
        raise ValueError(f"remote_dir must not contain '..': {remote_dir!r}")
    root = REMOTE_ROOT.rstrip("/")
    if normalized != root and not normalized.startswith(root + "/"):
        raise ValueError(f"remote_dir must be under {REMOTE_ROOT}: {remote_dir!r}")


def parse_env(items: list[str] | tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"env must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"invalid env key: {key!r}")
        parsed.append((key, value))
    return tuple(parsed)


def resolve_script(local_dir: Path, script: str) -> str:
    root = local_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"local_dir does not exist or is not a directory: {local_dir}")
    script_path = Path(script)
    resolved = script_path.resolve() if script_path.is_absolute() else (root / script_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"script does not exist or is not a file: {script}")
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"script must be inside local_dir: {script}")
    return resolved.relative_to(root).as_posix()


def build_plan(
    *,
    host: HostSpec,
    local_dir: Path,
    script: str,
    session: str | None = None,
    remote_dir: str | None = None,
    log_path: str | None = None,
    window: str | None = None,
    args: list[str] | tuple[str, ...] = (),
    env: list[str] | tuple[str, ...] = (),
    replace: bool = False,
    new_session: bool = False,
) -> LaunchPlan:
    script_rel = resolve_script(local_dir, script)
    generated_session = session or utc_run_id(script_rel)
    validate_name(generated_session, "session")
    if session is not None:
        validate_name(session, "session")
    window_name = window or Path(script_rel).stem or generated_session
    validate_name(window_name, "window")
    final_remote_dir = remote_dir or f"{REMOTE_ROOT}/{generated_session}"
    validate_remote_dir(final_remote_dir)
    final_log_path = log_path or posixpath.join(final_remote_dir, "run.log")
    if not final_log_path.startswith("/"):
        raise ValueError(f"log path must be absolute: {final_log_path!r}")
    return LaunchPlan(
        host=host,
        local_dir=local_dir.resolve(),
        script_rel=script_rel,
        remote_dir=posixpath.normpath(final_remote_dir),
        log_path=posixpath.normpath(final_log_path),
        generated_session=generated_session,
        requested_session=session,
        window=window_name,
        args=tuple(args),
        env=parse_env(env),
        replace=replace,
        new_session=new_session,
    )


def build_upload_argv(host: HostSpec, remote_dir: str, *, replace: bool = False) -> list[str]:
    validate_remote_dir(remote_dir)
    remote_q = shlex.quote(remote_dir)
    if replace:
        command = f"set -euo pipefail; rm -rf {remote_q}; mkdir -p {remote_q}; tar -xf - -C {remote_q}"
    else:
        command = (
            "set -euo pipefail; "
            f"if [ -d {remote_q} ] && [ \"$(find {remote_q} -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)\" ]; "
            f"then echo 'ERROR: remote_dir exists and is non-empty: {remote_dir}' >&2; exit 17; fi; "
            f"mkdir -p {remote_q}; tar -xf - -C {remote_q}"
        )
    return ["ssh", host.ssh_host, "bash", "-lc", command]


def build_launcher_source(plan: LaunchPlan) -> str:
    env_lines = [f"export {key}={shlex.quote(value)}" for key, value in plan.env]
    script_parts = ["bash", plan.script_rel, *plan.args]
    script_cmd = " ".join(shlex.quote(part) for part in script_parts)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(plan.remote_dir)}",
        *env_lines,
        f"{script_cmd} 2>&1 | tee -a {shlex.quote(plan.log_path)}",
        "",
    ]
    return "\n".join(lines)


def build_write_launcher_argv(host: HostSpec, remote_dir: str) -> list[str]:
    validate_remote_dir(remote_dir)
    path = posixpath.join(remote_dir, ".ucl_launch.sh")
    path_q = shlex.quote(path)
    command = f"set -euo pipefail; cat > {path_q}; chmod +x {path_q}"
    return ["ssh", host.ssh_host, "bash", "-lc", command]


def build_tmux_list_argv(host: HostSpec) -> list[str]:
    return ["ssh", host.ssh_host, "bash", "-lc", "tmux list-sessions -F '#{session_name}' 2>/dev/null || true"]


def parse_tmux_sessions(stdout: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in stdout.splitlines() if line.strip())


def decide_tmux(
    *,
    sessions: tuple[str, ...],
    generated_session: str,
    requested_session: str | None,
    new_session: bool,
    window: str,
) -> TmuxDecision:
    validate_name(generated_session, "session")
    validate_name(window, "window")
    if requested_session is not None:
        validate_name(requested_session, "session")
    if new_session:
        target = requested_session or generated_session
        if target in sessions:
            raise RuntimeError(f"tmux session already exists: {target}")
        return TmuxDecision("new-session", target, window, sessions)
    if requested_session is not None:
        if requested_session in sessions:
            return TmuxDecision("new-window", requested_session, window, sessions)
        return TmuxDecision("new-session", requested_session, window, sessions)
    if len(sessions) == 0:
        return TmuxDecision("new-session", generated_session, window, sessions)
    if len(sessions) == 1:
        return TmuxDecision("new-window", sessions[0], window, sessions)
    raise RuntimeError("multiple tmux sessions exist; pass --session or --new-session: " + ", ".join(sessions))


def build_tmux_launch_argv(host: HostSpec, plan: LaunchPlan, decision: TmuxDecision) -> list[str]:
    launcher = shlex.quote(posixpath.join(plan.remote_dir, ".ucl_launch.sh"))
    session_q = shlex.quote(decision.session)
    window_q = shlex.quote(decision.window)
    if decision.mode == "new-session":
        command = f"tmux new-session -d -s {session_q} {launcher}"
    elif decision.mode == "new-window":
        command = f"tmux new-window -d -t {session_q} -n {window_q} {launcher}"
    else:
        raise ValueError(f"unknown tmux mode: {decision.mode}")
    return ["ssh", host.ssh_host, "bash", "-lc", command]


def upload_bundle(
    plan: LaunchPlan,
    *,
    runner: Runner = subprocess.run,
    popener: Popener = subprocess.Popen,
) -> None:
    tar_proc = popener(["tar", "-cf", "-", "-C", str(plan.local_dir), "."], stdout=subprocess.PIPE)
    try:
        proc = runner(
            build_upload_argv(plan.host, plan.remote_dir, replace=plan.replace),
            stdin=tar_proc.stdout,
            capture_output=True,
            shell=False,
        )
        if tar_proc.stdout is not None:
            tar_proc.stdout.close()
        tar_returncode = tar_proc.wait()
    finally:
        if tar_proc.stdout is not None and not tar_proc.stdout.closed:
            tar_proc.stdout.close()
    if tar_returncode != 0:
        raise RuntimeError(f"local tar failed with exit {tar_returncode}")
    if int(getattr(proc, "returncode", 1)) != 0:
        stderr = (getattr(proc, "stderr", b"") or b"").decode(errors="replace").strip()
        stdout = (getattr(proc, "stdout", b"") or b"").decode(errors="replace").strip()
        raise RuntimeError(stderr or stdout or f"remote upload failed with exit {getattr(proc, 'returncode', 'unknown')}")


def write_launcher(plan: LaunchPlan, *, runner: Runner = subprocess.run) -> None:
    proc = runner(
        build_write_launcher_argv(plan.host, plan.remote_dir),
        input=build_launcher_source(plan),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "failed to write launcher").strip())


def list_remote_sessions(host: HostSpec, *, runner: Runner = subprocess.run) -> tuple[str, ...]:
    proc = runner(build_tmux_list_argv(host), capture_output=True, text=True, shell=False)
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or "failed to list tmux sessions").strip())
    return parse_tmux_sessions(getattr(proc, "stdout", "") or "")


def launch_tmux(plan: LaunchPlan, decision: TmuxDecision, *, runner: Runner = subprocess.run) -> None:
    proc = runner(build_tmux_launch_argv(plan.host, plan, decision), capture_output=True, text=True, shell=False)
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "failed to launch tmux").strip())


def format_summary(plan: LaunchPlan, decision: TmuxDecision) -> str:
    return "\n".join(
        [
            f"host:       {plan.host.name}",
            f"session:    {decision.session}",
            f"window:     {decision.window}",
            f"remote_dir: {plan.remote_dir}",
            f"log:        {plan.log_path}",
            "",
            "attach:",
            f"  ssh {plan.host.ssh_host}",
            f"  tmux attach -t {decision.session}",
            "",
            "tail:",
            f"  ssh {plan.host.ssh_host} 'tail -f {plan.log_path}'",
        ]
    )
