"""Remote script planning, upload, and tmux execution primitives."""

from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.profiles import (
    ResolvedProfile,
    bash_export_lines,
    csh_setenv_lines,
    parse_cli_env,
    shell_join,
    validate_name,
)


Runner = Callable[..., subprocess.CompletedProcess]
Popener = Callable[..., subprocess.Popen]
REMOTE_ROOT = "/tmp/ucl-machine-tools/launchers"
TMUX_SENTINEL_BEGIN = "UCL_TMUX_JSON_BEGIN"
TMUX_SENTINEL_END = "UCL_TMUX_JSON_END"


def remote_bash_argv(host: HostSpec, command: str) -> list[str]:
    if "'" in command:
        raise ValueError("remote bash command must not contain single quotes; use stdin for complex scripts")
    return ["ssh", host.ssh_host, "bash", "-lc", f"'{command}'"]


@dataclass(frozen=True)
class RemoteJobPlan:
    kind: str
    host: HostSpec
    run_id: str
    remote_dir: str
    log_path: str
    command: tuple[str, ...]
    profile: ResolvedProfile
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


def utc_run_id(stem: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._-") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe}_{timestamp}"


def validate_remote_dir(remote_dir: str) -> None:
    if not remote_dir.startswith("/"):
        raise ValueError(f"remote_dir must be absolute: {remote_dir!r}")
    normalized = posixpath.normpath(remote_dir)
    if normalized == "/" or ".." in normalized.split("/"):
        raise ValueError(f"remote_dir must not contain '..': {remote_dir!r}")
    root = REMOTE_ROOT.rstrip("/")
    if normalized != root and not normalized.startswith(root + "/"):
        raise ValueError(f"remote_dir must be under {REMOTE_ROOT}: {remote_dir!r}")


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


def build_run_plan(
    *,
    host: HostSpec,
    local_dir: Path,
    script: str,
    profile: ResolvedProfile,
    args: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    session: str | None = None,
    new_session: bool = False,
    window: str | None = None,
    remote_dir: str | None = None,
    log_path: str | None = None,
    replace: bool = False,
) -> RemoteJobPlan:
    script_rel = resolve_script(local_dir, script)
    command = (*profile.run_prefix, "bash", script_rel, *args)
    run_id = session or utc_run_id(Path(script_rel).stem or "run")
    validate_name(run_id, "run_id")
    if session is not None:
        validate_name(session, "session")
    window_name = window or Path(script_rel).stem or run_id
    validate_name(window_name, "window")
    final_remote_dir = posixpath.normpath(remote_dir or f"{REMOTE_ROOT}/{run_id}")
    validate_remote_dir(final_remote_dir)
    final_log_path = posixpath.normpath(log_path or posixpath.join(final_remote_dir, "run.log"))
    if not final_log_path.startswith("/"):
        raise ValueError(f"log path must be absolute: {final_log_path!r}")
    if env:
        # CLI env has already been merged into the profile by the caller.
        parse_cli_env(env)
    return RemoteJobPlan(
        kind="run",
        host=host,
        run_id=run_id,
        remote_dir=final_remote_dir,
        log_path=final_log_path,
        command=command,
        profile=profile,
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
    profile: ResolvedProfile,
    session: str | None = None,
    new_session: bool = False,
    window: str | None = None,
    remote_dir: str | None = None,
    log_path: str | None = None,
) -> RemoteJobPlan:
    if bool(command) == bool(stdin_body is not None):
        raise ValueError("exec requires exactly one of command tokens or --stdin")
    stem = command[0] if command else "stdin"
    run_id = utc_run_id(f"exec_{stem}")
    if session is not None:
        validate_name(session, "session")
    window_name = window or f"exec_{stem}"
    validate_name(window_name, "window")
    final_remote_dir = posixpath.normpath(remote_dir or f"{REMOTE_ROOT}/{run_id}")
    validate_remote_dir(final_remote_dir)
    final_log_path = posixpath.normpath(log_path or posixpath.join(final_remote_dir, "run.log"))
    if not final_log_path.startswith("/"):
        raise ValueError(f"log path must be absolute: {final_log_path!r}")
    final_command = (*profile.run_prefix, *command) if command else ()
    return RemoteJobPlan(
        kind="exec",
        host=host,
        run_id=run_id,
        remote_dir=final_remote_dir,
        log_path=final_log_path,
        command=final_command,
        profile=profile,
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
        f"echo {shlex.quote('[ucl] profile: ' + ','.join(plan.profile.names))}",
        *bash_export_lines(plan.profile.env),
    ]
    for check in (*plan.profile.preflight, *plan.profile.preflight_after_setup):
        lines.append(f"echo {shlex.quote('[ucl] preflight: ' + check.label)}")
        lines.append(check.command)
    lines.append("echo '[ucl] run'")
    if plan.stdin_body is not None:
        if plan.profile.run_prefix:
            lines.append(shell_join((*plan.profile.run_prefix, "bash")) + " <<'UCL_STDIN_SCRIPT'")
            lines.append(plan.stdin_body)
            lines.append("UCL_STDIN_SCRIPT")
        else:
            lines.append(plan.stdin_body)
    else:
        lines.append(shell_join(plan.command))
    lines.append("")
    return "\n".join(lines)


def _csh_launcher_source(plan: RemoteJobPlan) -> str:
    bash_path = posixpath.join(plan.remote_dir, ".ucl_payload.sh")
    lines = [
        "#!/bin/csh -f",
        f"cd {shlex.quote(plan.remote_dir)}",
    ]
    for source_path in plan.profile.source:
        lines.append(f"source {shlex.quote(source_path)}")
    lines.extend(csh_setenv_lines(plan.profile.env))
    lines.append(f"exec bash {shlex.quote(bash_path)}")
    lines.append("")
    return "\n".join(lines)


def build_launcher_files(plan: RemoteJobPlan) -> tuple[str, dict[str, str]]:
    bash_source = _bash_payload_source(plan)
    if plan.profile.shell == "csh-bootstrap":
        return ".ucl_launch.csh", {
            ".ucl_payload.sh": bash_source,
            ".ucl_launch.csh": _csh_launcher_source(plan),
        }
    return ".ucl_payload.sh", {".ucl_payload.sh": bash_source}


def build_remote_mkdir_command(remote_dir: str, *, replace: bool = False) -> str:
    validate_remote_dir(remote_dir)
    remote_q = shlex.quote(remote_dir)
    if replace:
        return f"set -euo pipefail; rm -rf {remote_q}; mkdir -p {remote_q}"
    return (
        "set -euo pipefail; "
        f"if [ -d {remote_q} ] && [ \"$(find {remote_q} -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)\" ]; "
        f"then echo ERROR: remote_dir exists and is non-empty: {remote_dir} >&2; exit 17; fi; "
        f"mkdir -p {remote_q}"
    )


def build_remote_mkdir_argv(host: HostSpec, remote_dir: str, *, replace: bool = False) -> list[str]:
    return remote_bash_argv(host, build_remote_mkdir_command(remote_dir, replace=replace))


def build_upload_argv(host: HostSpec, remote_dir: str, *, replace: bool = False) -> list[str]:
    mkdir = build_remote_mkdir_command(remote_dir, replace=replace)
    remote_q = shlex.quote(remote_dir)
    return remote_bash_argv(host, f"{mkdir}; tar -xf - -C {remote_q}")


def build_write_file_argv(host: HostSpec, remote_dir: str, name: str) -> list[str]:
    validate_remote_dir(remote_dir)
    if "/" in name or not name:
        raise ValueError(f"remote file name must be a basename: {name!r}")
    path = posixpath.join(remote_dir, name)
    command = f"set -euo pipefail; cat > {shlex.quote(path)}; chmod +x {shlex.quote(path)}"
    return remote_bash_argv(host, command)


def build_tmux_list_argv(host: HostSpec) -> list[str]:
    return ["ssh", host.ssh_host, "python3", "-"]


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
    session_q = shlex.quote(decision.session)
    window_q = shlex.quote(decision.window)
    launcher_q = '"' + launcher.replace('"', '\\"') + '"'
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


def create_remote_dir(plan: RemoteJobPlan, *, runner: Runner = subprocess.run) -> None:
    proc = runner(build_remote_mkdir_argv(plan.host, plan.remote_dir, replace=plan.replace), capture_output=True, text=True, shell=False)
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "failed to create remote dir").strip())


def write_launcher_files(plan: RemoteJobPlan, *, runner: Runner = subprocess.run) -> str:
    launcher_name, files = build_launcher_files(plan)
    for name, source in files.items():
        proc = runner(
            build_write_file_argv(plan.host, plan.remote_dir, name),
            input=source,
            capture_output=True,
            text=True,
            shell=False,
        )
        if int(getattr(proc, "returncode", 1)) != 0:
            raise RuntimeError((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or f"failed to write {name}").strip())
    return launcher_name


def list_remote_sessions(host: HostSpec, *, runner: Runner = subprocess.run) -> tuple[str, ...]:
    proc = runner(
        build_tmux_list_argv(host),
        input=build_tmux_list_source(),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", "") or "failed to list tmux sessions").strip())
    return parse_tmux_sessions(getattr(proc, "stdout", "") or "")


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
            f"  ucl tail {plan.run_id}",
        ]
    )
