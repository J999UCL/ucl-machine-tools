"""Remote script planning, upload, and tmux execution primitives."""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.ssh import build_remote_python_argv, describe_ssh_failure


Runner = Callable[..., subprocess.CompletedProcess]
Popener = Callable[..., subprocess.Popen]
DEFAULT_REMOTE_ROOT = "/tmp/ucl-machine-tools/launchers"
REMOTE_ROOT = DEFAULT_REMOTE_ROOT
REMOTE_ROOT_ENV = "UCL_LAUNCH_ROOT"
TMUX_SENTINEL_BEGIN = "UCL_TMUX_JSON_BEGIN"
TMUX_SENTINEL_END = "UCL_TMUX_JSON_END"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_SHELLS = {"bash", "csh"}


@dataclass(frozen=True)
class RemoteJobPlan:
    kind: str
    host: HostSpec
    run_id: str
    remote_dir: str
    remote_root: str
    log_path: str
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    shell: str
    requested_session: str | None
    new_session: bool
    window: str
    local_dir: Path | None = None
    replace: bool = False
    stdin_body: str | None = None


@dataclass(frozen=True)
class TmuxDecision:
    mode: str
    session: str
    window: str
    existing_sessions: tuple[str, ...]


def remote_bash_argv(host: HostSpec, command: str) -> list[str]:
    if "'" in command:
        raise ValueError("remote bash command must not contain single quotes; use stdin for complex scripts")
    return ["ssh", host.ssh_host, "bash", "-lc", f"'{command}'"]


def validate_name(value: str, label: str) -> None:
    if not value or not SAFE_NAME_RE.match(value):
        raise ValueError(f"{label} may only contain letters, numbers, dot, dash, and underscore: {value!r}")


def validate_shell(shell: str) -> None:
    if shell not in SUPPORTED_SHELLS:
        raise ValueError(f"shell must be one of {sorted(SUPPORTED_SHELLS)}: {shell!r}")


def parse_env(items: Iterable[str]) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"env must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not ENV_KEY_RE.match(key):
            raise ValueError(f"invalid env key: {key!r}")
        if "\x00" in value:
            raise ValueError(f"env value for {key!r} must not contain NUL bytes")
        parsed.append((key, value))
    return tuple(parsed)


def shell_join(tokens: Iterable[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def bash_export_lines(env: Iterable[tuple[str, str]]) -> list[str]:
    return [f"export {key}={shlex.quote(value)}" for key, value in env]


def csh_setenv_lines(env: Iterable[tuple[str, str]]) -> list[str]:
    return [f"setenv {key} {shlex.quote(value)}" for key, value in env]


def utc_run_id(stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._-") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe}_{timestamp}"


def default_remote_root() -> str:
    return posixpath.normpath(os.environ.get(REMOTE_ROOT_ENV, DEFAULT_REMOTE_ROOT))


def validate_remote_root(remote_root: str) -> str:
    if not remote_root.startswith("/"):
        raise ValueError(f"remote_root must be absolute: {remote_root!r}")
    normalized = posixpath.normpath(remote_root)
    if normalized == "/" or ".." in normalized.split("/"):
        raise ValueError(f"remote_root must not contain '..': {remote_root!r}")
    return normalized


def validate_remote_dir(remote_dir: str, *, remote_root: str | None = None) -> None:
    if not remote_dir.startswith("/"):
        raise ValueError(f"remote_dir must be absolute: {remote_dir!r}")
    normalized = posixpath.normpath(remote_dir)
    if normalized == "/" or ".." in normalized.split("/"):
        raise ValueError(f"remote_dir must not contain '..': {remote_dir!r}")
    root = validate_remote_root(remote_root or default_remote_root()).rstrip("/")
    if normalized != root and not normalized.startswith(root + "/"):
        raise ValueError(f"remote_dir must be under {root}: {remote_dir!r}")


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


def _common_plan_values(
    *,
    stem: str,
    session: str | None,
    window: str | None,
    remote_dir: str | None,
    remote_root: str | None,
    log_path: str | None,
) -> tuple[str, str, str, str, str]:
    run_id = session or utc_run_id(stem)
    validate_name(run_id, "run_id")
    if session is not None:
        validate_name(session, "session")
    window_name = window or stem or run_id
    validate_name(window_name, "window")
    final_remote_root = validate_remote_root(remote_root or default_remote_root())
    final_remote_dir = posixpath.normpath(remote_dir or f"{final_remote_root}/{run_id}")
    validate_remote_dir(final_remote_dir, remote_root=final_remote_root)
    final_log_path = posixpath.normpath(log_path or posixpath.join(final_remote_dir, "run.log"))
    if not final_log_path.startswith("/"):
        raise ValueError(f"log path must be absolute: {final_log_path!r}")
    return run_id, window_name, final_remote_dir, final_remote_root, final_log_path


def build_run_plan(
    *,
    host: HostSpec,
    local_dir: Path,
    script: str,
    args: tuple[str, ...] = (),
    env: tuple[tuple[str, str], ...] = (),
    shell: str = "bash",
    session: str | None = None,
    new_session: bool = False,
    window: str | None = None,
    remote_dir: str | None = None,
    remote_root: str | None = None,
    log_path: str | None = None,
    replace: bool = False,
) -> RemoteJobPlan:
    validate_shell(shell)
    script_rel = resolve_script(local_dir, script)
    run_id, window_name, final_remote_dir, final_remote_root, final_log_path = _common_plan_values(
        stem=Path(script_rel).stem or "run",
        session=session,
        window=window,
        remote_dir=remote_dir,
        remote_root=remote_root,
        log_path=log_path,
    )
    return RemoteJobPlan(
        kind="run",
        host=host,
        run_id=run_id,
        remote_dir=final_remote_dir,
        remote_root=final_remote_root,
        log_path=final_log_path,
        command=("bash", script_rel, *args),
        env=env,
        shell=shell,
        requested_session=session,
        new_session=new_session,
        window=window_name,
        local_dir=local_dir.resolve(),
        replace=replace,
    )


def build_exec_plan(
    *,
    host: HostSpec,
    command: tuple[str, ...] = (),
    stdin_body: str | None = None,
    env: tuple[tuple[str, str], ...] = (),
    shell: str = "bash",
    session: str | None = None,
    new_session: bool = False,
    window: str | None = None,
    remote_dir: str | None = None,
    remote_root: str | None = None,
    log_path: str | None = None,
) -> RemoteJobPlan:
    validate_shell(shell)
    if bool(command) == bool(stdin_body is not None):
        raise ValueError("exec requires exactly one of command tokens or --stdin")
    stem = f"exec_{command[0] if command else 'stdin'}"
    run_id, window_name, final_remote_dir, final_remote_root, final_log_path = _common_plan_values(
        stem=stem,
        session=session,
        window=window,
        remote_dir=remote_dir,
        remote_root=remote_root,
        log_path=log_path,
    )
    return RemoteJobPlan(
        kind="exec",
        host=host,
        run_id=run_id,
        remote_dir=final_remote_dir,
        remote_root=final_remote_root,
        log_path=final_log_path,
        command=command,
        env=env,
        shell=shell,
        requested_session=session,
        new_session=new_session,
        window=window_name,
        stdin_body=stdin_body,
    )


def _bash_payload_source(plan: RemoteJobPlan) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        f"cd {shlex.quote(plan.remote_dir)}",
        f"mkdir -p {shlex.quote(posixpath.dirname(plan.log_path))}",
        f"exec > >(tee -a {shlex.quote(plan.log_path)}) 2>&1",
        "trap 'rc=$?; echo \"[ucl] failed rc=$rc line=$LINENO\"; exit $rc' ERR",
        "echo '[ucl] shell: bash'",
        *bash_export_lines(plan.env),
        "echo '[ucl] run'",
    ]
    lines.append(plan.stdin_body if plan.stdin_body is not None else shell_join(plan.command))
    lines.append("")
    return "\n".join(lines)


def _bash_csh_launcher_source(plan: RemoteJobPlan) -> str:
    csh_path = posixpath.join(plan.remote_dir, ".ucl_payload.csh")
    lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        f"cd {shlex.quote(plan.remote_dir)}",
        f"mkdir -p {shlex.quote(posixpath.dirname(plan.log_path))}",
        f"exec > >(tee -a {shlex.quote(plan.log_path)}) 2>&1",
        "trap 'rc=$?; echo \"[ucl] failed rc=$rc line=$LINENO\"; exit $rc' ERR",
        "echo '[ucl] shell: csh'",
        f"csh -f {shlex.quote(csh_path)}",
        "",
    ]
    return "\n".join(lines)


def _csh_payload_source(plan: RemoteJobPlan) -> str:
    lines = [
        "#!/bin/csh -f",
        f"cd {shlex.quote(plan.remote_dir)}",
        *csh_setenv_lines(plan.env),
        "echo '[ucl] run'",
    ]
    lines.append(plan.stdin_body if plan.stdin_body is not None else shell_join(plan.command))
    lines.append("")
    return "\n".join(lines)


def build_launcher_files(plan: RemoteJobPlan) -> tuple[str, dict[str, str]]:
    if plan.shell == "csh":
        return ".ucl_launch.sh", {
            ".ucl_launch.sh": _bash_csh_launcher_source(plan),
            ".ucl_payload.csh": _csh_payload_source(plan),
        }
    return ".ucl_payload.sh", {".ucl_payload.sh": _bash_payload_source(plan)}


def build_remote_mkdir_command(remote_dir: str, *, remote_root: str | None = None, replace: bool = False) -> str:
    validate_remote_dir(remote_dir, remote_root=remote_root)
    remote_q = shlex.quote(remote_dir)
    if replace:
        return f"set -euo pipefail; rm -rf {remote_q}; mkdir -p {remote_q}"
    return (
        "set -euo pipefail; "
        f"if [ -d {remote_q} ] && [ \"$(find {remote_q} -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)\" ]; "
        f"then echo ERROR: remote_dir exists and is non-empty: {remote_dir} >&2; exit 17; fi; "
        f"mkdir -p {remote_q}"
    )


def build_remote_mkdir_argv(host: HostSpec, remote_dir: str, *, remote_root: str | None = None, replace: bool = False) -> list[str]:
    return remote_bash_argv(host, build_remote_mkdir_command(remote_dir, remote_root=remote_root, replace=replace))


def build_upload_argv(host: HostSpec, remote_dir: str, *, remote_root: str | None = None, replace: bool = False) -> list[str]:
    mkdir = build_remote_mkdir_command(remote_dir, remote_root=remote_root, replace=replace)
    remote_q = shlex.quote(remote_dir)
    return remote_bash_argv(host, f"{mkdir}; tar -xf - -C {remote_q}")


def build_write_file_argv(host: HostSpec, remote_dir: str, name: str, *, remote_root: str | None = None) -> list[str]:
    validate_remote_dir(remote_dir, remote_root=remote_root)
    if "/" in name or not name:
        raise ValueError(f"remote file name must be a basename: {name!r}")
    path = posixpath.join(remote_dir, name)
    command = f"set -euo pipefail; cat > {shlex.quote(path)}; chmod +x {shlex.quote(path)}"
    return remote_bash_argv(host, command)


def build_tmux_list_argv(host: HostSpec, *, timeout_seconds: int | None = None) -> list[str]:
    return build_remote_python_argv(host.ssh_host, timeout_seconds=timeout_seconds)


def build_tmux_list_source() -> str:
    return f"""
import json
import subprocess
BEGIN={json.dumps(TMUX_SENTINEL_BEGIN)}
END={json.dumps(TMUX_SENTINEL_END)}
proc = subprocess.run(["tmux", "list-sessions", "-F", "#{{session_name}}"], capture_output=True, text=True)
sessions = [] if proc.returncode != 0 else [line.strip() for line in proc.stdout.splitlines() if line.strip()]
print(BEGIN)
print(json.dumps({{"schema_version": 1, "sessions": sessions}}, sort_keys=True))
print(END)
"""


def parse_tmux_sessions(stdout: str) -> tuple[str, ...]:
    payloads: list[str] = []
    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != TMUX_SENTINEL_BEGIN:
            continue
        for end in range(idx + 1, len(lines)):
            if lines[end].strip() == TMUX_SENTINEL_END:
                payloads.append("\n".join(lines[idx + 1 : end]).strip())
                break
    if not payloads:
        raise ValueError("tmux sentinel not found")
    if len(payloads) > 1:
        raise ValueError("multiple tmux sentinels found")
    payload = json.loads(payloads[0])
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sessions"), list):
        raise ValueError("invalid tmux sentinel payload")
    return tuple(str(item) for item in payload["sessions"])


def decide_tmux(
    *,
    sessions: tuple[str, ...],
    generated_session: str,
    requested_session: str | None,
    new_session: bool,
    window: str,
    require_explicit_when_not_single: bool = False,
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
    if len(sessions) == 1:
        return TmuxDecision("new-window", sessions[0], window, sessions)
    if require_explicit_when_not_single:
        if not sessions:
            raise RuntimeError("no tmux sessions exist; pass --session NAME or --new-session")
        raise RuntimeError("multiple tmux sessions exist; pass --session or --new-session: " + ", ".join(sessions))
    if len(sessions) == 0:
        return TmuxDecision("new-session", generated_session, window, sessions)
    raise RuntimeError("multiple tmux sessions exist; pass --session or --new-session: " + ", ".join(sessions))


def build_tmux_launch_argv(host: HostSpec, plan: RemoteJobPlan, decision: TmuxDecision, launcher_name: str) -> list[str]:
    launcher_path = shlex.quote(posixpath.join(plan.remote_dir, launcher_name))
    launcher = f"csh -f {launcher_path}" if launcher_name.endswith(".csh") else f"bash {launcher_path}"
    launcher_q = '"' + launcher.replace('"', '\\"') + '"'
    session_q = shlex.quote(decision.session)
    window_q = shlex.quote(decision.window)
    if decision.mode == "new-session":
        command = f"tmux new-session -d -s {session_q} {launcher_q}"
    elif decision.mode == "new-window":
        command = f"tmux new-window -d -t {session_q} -n {window_q} {launcher_q}"
    else:
        raise ValueError(f"unknown tmux mode: {decision.mode}")
    return remote_bash_argv(host, command)


def upload_bundle(plan: RemoteJobPlan, *, runner: Runner = subprocess.run, popener: Popener = subprocess.Popen) -> None:
    if plan.local_dir is None:
        raise ValueError("upload_bundle requires local_dir")
    tar_proc = popener(["tar", "-cf", "-", "-C", str(plan.local_dir), "."], stdout=subprocess.PIPE)
    try:
        proc = runner(
            build_upload_argv(plan.host, plan.remote_dir, remote_root=plan.remote_root, replace=plan.replace),
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


def create_remote_dir(plan: RemoteJobPlan, *, runner: Runner = subprocess.run) -> None:
    proc = runner(build_remote_mkdir_argv(plan.host, plan.remote_dir, remote_root=plan.remote_root, replace=plan.replace), capture_output=True, text=True, shell=False)
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "failed to create remote dir").strip())


def write_launcher_files(plan: RemoteJobPlan, *, runner: Runner = subprocess.run) -> str:
    launcher_name, files = build_launcher_files(plan)
    for name, source in files.items():
        proc = runner(
            build_write_file_argv(plan.host, plan.remote_dir, name, remote_root=plan.remote_root),
            input=source,
            capture_output=True,
            text=True,
            shell=False,
        )
        if int(getattr(proc, "returncode", 1)) != 0:
            raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or f"failed to write {name}").strip())
    return launcher_name


def list_remote_sessions(
    host: HostSpec,
    *,
    runner: Runner = subprocess.run,
    timeout_seconds: int | None = None,
) -> tuple[str, ...]:
    try:
        proc = runner(
            build_tmux_list_argv(host, timeout_seconds=timeout_seconds),
            input=build_tmux_list_source(),
            capture_output=True,
            text=True,
            timeout=(timeout_seconds + 3) if timeout_seconds is not None else None,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timed out listing tmux sessions on {host.name} after {exc.timeout}s") from exc
    if int(getattr(proc, "returncode", 1)) != 0:
        returncode = int(getattr(proc, "returncode", 1))
        detail = describe_ssh_failure(
            returncode,
            stdout=getattr(proc, "stdout", "") or "",
            stderr=getattr(proc, "stderr", "") or "",
        )
        raise RuntimeError(f"failed to list tmux sessions on {host.name}: {detail} (exit {returncode})")
    try:
        return parse_tmux_sessions(getattr(proc, "stdout", "") or "")
    except ValueError as exc:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        stdout = (getattr(proc, "stdout", "") or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(f"failed to parse tmux sessions on {host.name}: {detail}") from exc


def launch_tmux(plan: RemoteJobPlan, decision: TmuxDecision, launcher_name: str, *, runner: Runner = subprocess.run) -> None:
    proc = runner(build_tmux_launch_argv(plan.host, plan, decision, launcher_name), capture_output=True, text=True, shell=False)
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "failed to launch tmux").strip())


def format_summary(plan: RemoteJobPlan, decision: TmuxDecision) -> str:
    return "\n".join(
        [
            f"host:       {plan.host.name}",
            f"run_id:     {plan.run_id}",
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
            f"  ucl tail {plan.run_id} --live",
        ]
    )
