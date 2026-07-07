"""Unified CLI for UCL machine tools."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from ucl_machine_tools.hosts import HostSpec, load_catalog, parse_selector
from ucl_machine_tools import inventory
from ucl_machine_tools.launch import (
    build_exec_plan,
    build_run_plan,
    create_remote_dir,
    decide_tmux,
    format_summary,
    launch_tmux,
    list_remote_sessions,
    upload_bundle,
    write_launcher_files,
)
from ucl_machine_tools.profiles import (
    ResolvedProfile,
    bash_export_lines,
    csh_setenv_lines,
    load_profiles,
    parse_cli_env,
    resolve_profiles,
    shell_join,
)
from ucl_machine_tools.registry import RunRecord, read_record, write_record
from ucl_machine_tools.ssh import ensure_knuckles_master


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified UCL machine helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Check GPU, scratch, and host state.")
    status.add_argument("items", nargs="*", help="optional mode and target, e.g. recommend 3090ti")
    _add_inventory_flags(status)

    doctor = subparsers.add_parser("doctor", help="Check one host and optional launch profile.")
    doctor.add_argument("host")
    doctor.add_argument("--catalog", type=Path)
    doctor.add_argument("--profile", action="append", default=[], help="profile to validate; repeatable")
    doctor.add_argument("--profile-file", action="append", default=[], type=Path)
    doctor.add_argument("--env", action="append", default=[])
    doctor.add_argument("--gpu", help="GPU id to expose while checking profile")
    doctor.add_argument("--timeout-seconds", type=int, default=8)

    run = subparsers.add_parser("run", help="Upload a local bundle and launch it in tmux.")
    run.add_argument("--host", required=True)
    run.add_argument("--local-dir", required=True, type=Path)
    run.add_argument("--script", required=True)
    _add_launch_common_flags(run)
    run.add_argument("--arg", action="append", default=[], help="script argument; repeat for multiple args")
    run.add_argument("--replace", action="store_true", help="replace an existing non-empty remote bundle dir")

    exec_parser = subparsers.add_parser("exec", help="Run a small remote command inside tmux.")
    exec_parser.add_argument("host")
    _add_launch_common_flags(exec_parser)
    exec_parser.add_argument("--stdin", action="store_true", help="read a bash script from stdin")

    tail = subparsers.add_parser("tail", help="Print or follow a recorded run log.")
    tail.add_argument("run_ref", nargs="?", default="last")
    tail.add_argument("--lines", type=int, default=80)
    tail.add_argument("--follow", action="store_true")

    fetch = subparsers.add_parser("fetch", help="Fetch small artifacts for a recorded run.")
    fetch.add_argument("run_ref", nargs="?", default="last")
    fetch.add_argument("--output-dir", type=Path)

    clean = subparsers.add_parser("clean", help="List or delete old launcher directories.")
    clean.add_argument("host")
    clean.add_argument("--catalog", type=Path)
    clean.add_argument("--execute", action="store_true", help="delete listed launcher directories")
    clean.add_argument("--older-than-days", type=int, default=7)

    return parser


def _add_inventory_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selector", help="explicit selector; overrides positional target")
    parser.add_argument("--catalog", type=Path, help="host catalog JSON")
    parser.add_argument("--root", default="/tmp/ucl-machine-tools", help="remote scratch root to check")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--table", action="store_true", help="emit a human table")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=8)
    parser.add_argument("--only-free", action="store_true")
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    parser.add_argument("--min-tmp-free-gb", type=float)
    parser.add_argument("--sizes", action="store_true")
    parser.add_argument("--debug", action="store_true")


def _add_launch_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--profile-file", action="append", default=[], type=Path)
    parser.add_argument("--env", action="append", default=[], help="remote env KEY=VALUE; repeat for multiple vars")
    parser.add_argument("--gpu", help="GPU id or auto")
    parser.add_argument("--session", help="tmux session to use or create")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session")
    parser.add_argument("--window", help="tmux window name")
    parser.add_argument("--remote-dir", help="remote bundle dir under /tmp/ucl-machine-tools/launchers")
    parser.add_argument("--log", help="remote log path; default is <remote-dir>/run.log")
    parser.add_argument("--dry-run", action="store_true")


def _resolve_one_host(selector: str, *, catalog_path: Path | None) -> HostSpec:
    catalog = load_catalog(catalog_path)
    hosts = parse_selector(selector, catalog=catalog)
    if len(hosts) != 1:
        raise ValueError(f"selector must resolve to exactly one host, got {len(hosts)} for {selector!r}")
    return hosts[0]


def _status_mode_and_target(items: list[str], selector: str | None) -> tuple[str, str]:
    if selector:
        return "check", selector
    if not items:
        return "check", "all"
    if items[0] in {"check", "gpus", "state", "recommend"}:
        return items[0], items[1] if len(items) > 1 else "all"
    return "check", items[0]


def _best_free_gpu(row: dict[str, Any], *, min_free_vram_gb: float) -> str:
    candidates: list[dict[str, Any]] = []
    for gpu in row.get("gpus", []) or []:
        if gpu.get("processes", []) or []:
            continue
        free_mb = gpu.get("memory_free_mb")
        if free_mb is None and gpu.get("memory_total_mb") is not None and gpu.get("memory_used_mb") is not None:
            free_mb = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        if free_mb is not None and float(free_mb) < min_free_vram_gb * 1024:
            continue
        util = gpu.get("utilization_gpu_percent")
        if util is not None and int(util) > 20:
            continue
        candidates.append(gpu)
    if not candidates:
        raise RuntimeError(f"no free GPU found on {row.get('host')}")
    best = max(candidates, key=lambda gpu: float(gpu.get("memory_free_mb") or 0))
    return str(best.get("index", 0))


def _resolve_profile(args: argparse.Namespace, *, extra_env: tuple[tuple[str, str], ...] = ()) -> ResolvedProfile:
    profile_catalog = load_profiles(explicit_files=args.profile_file)
    cli_env = (*parse_cli_env(args.env), *extra_env)
    return resolve_profiles(args.profile, catalog=profile_catalog, cli_env=cli_env)


def _gpu_env(
    args: argparse.Namespace,
    host: HostSpec,
    *,
    runner,
    min_free_vram_gb: float = 4.0,
) -> tuple[tuple[str, str], ...]:
    if not getattr(args, "gpu", None):
        return ()
    if args.gpu != "auto":
        return (("CUDA_VISIBLE_DEVICES", str(args.gpu)),)
    rows = inventory.collect([host], runner=runner, jobs=1, min_free_vram_gb=min_free_vram_gb)
    gpu_id = _best_free_gpu(rows[0], min_free_vram_gb=min_free_vram_gb)
    return (("CUDA_VISIBLE_DEVICES", gpu_id),)


def _filter_status_rows(mode: str, args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = rows
    if mode == "gpus":
        out = [row for row in out if row.get("gpus")]
    if mode == "state":
        out = [row for row in out if row.get("scratch") or row.get("status") not in {"no-gpu", "busy", "ready"}]
    if args.only_free or mode == "recommend":
        out = [row for row in out if row.get("status") == "ready"]
    if mode == "recommend":
        out = sorted(out, key=lambda row: _best_vram_mb(row), reverse=True)[:1]
    return out


def _best_vram_mb(row: dict[str, Any]) -> float:
    best = 0.0
    for gpu in row.get("gpus", []) or []:
        free = gpu.get("memory_free_mb")
        if free is None and gpu.get("memory_total_mb") is not None and gpu.get("memory_used_mb") is not None:
            free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        best = max(best, float(free or 0))
    return best


def run_status(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    mode, target = _status_mode_and_target(args.items, args.selector)
    catalog = load_catalog(args.catalog)
    selected = parse_selector(target, catalog=catalog)
    ensure_knuckles_master(runner=runner)
    rows = inventory.collect(
        selected,
        runner=runner,
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
        root=args.root,
        sizes=args.sizes,
        debug=args.debug,
        min_tmp_free_gb=float(args.min_tmp_free_gb) if args.min_tmp_free_gb is not None else 50.0,
        min_free_vram_gb=float(args.min_free_vram_gb),
    )
    rows = _filter_status_rows(mode, args, rows)
    if args.json and not args.table:
        print(json.dumps(inventory.to_jsonable(rows), indent=2, sort_keys=True))
    else:
        print(inventory.format_table(rows))
    return 0


def _profile_check_script(profile: ResolvedProfile, gpu: str | None) -> tuple[list[str], str]:
    env = profile.env
    if gpu:
        env = (*env, ("CUDA_VISIBLE_DEVICES", gpu))
    bash_lines = [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        *bash_export_lines(env),
    ]
    for check in (*profile.preflight, *profile.preflight_after_setup):
        bash_lines.append(f"echo {shlex.quote('[ucl] preflight: ' + check.label)}")
        bash_lines.append(check.command)
    bash_lines.append("echo '[ucl] profile check ok'")
    bash_body = "\n".join(bash_lines) + "\n"
    if profile.shell != "csh-bootstrap":
        return ["bash", "-s"], bash_body
    csh_lines = ["#!/bin/csh -f"]
    for source_path in profile.source:
        csh_lines.append(f"source {shlex.quote(source_path)}")
    csh_lines.extend(csh_setenv_lines(env))
    csh_lines.append("exec bash -s <<'UCL_PROFILE_BASH'")
    csh_lines.append(bash_body)
    csh_lines.append("UCL_PROFILE_BASH")
    return ["csh", "-f"], "\n".join(csh_lines) + "\n"


def run_doctor(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    profile = _resolve_profile(args)
    ensure_knuckles_master(runner=runner)
    row = inventory.collect([host], runner=runner, jobs=1, timeout_seconds=args.timeout_seconds)[0]
    sessions = list_remote_sessions(host, runner=runner)
    argv_tail, script = _profile_check_script(profile, args.gpu)
    proc = runner(["ssh", host.ssh_host, *argv_tail], input=script, capture_output=True, text=True, shell=False)
    profile_ok = int(getattr(proc, "returncode", 1)) == 0
    print(f"host:          {host.name}")
    print(f"status:        {row.get('status')}")
    print(f"tmp_scratch:  {'yes' if (row.get('scratch') or {}).get('exists') else 'no'}")
    print(f"tmux_sessions: {', '.join(sessions) if sessions else 'none'}")
    print(f"profile:       {','.join(profile.names)}")
    print(f"profile_check: {'ok' if profile_ok else 'failed'}")
    if not profile_ok:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        if detail:
            print(detail)
        return 2
    return 0


def _strip_remainder(command: list[str]) -> tuple[str, ...]:
    if command and command[0] == "--":
        command = command[1:]
    return tuple(command)


def _dry_run_summary(plan, *, subcommand: str) -> str:
    return "\n".join(
        [
            "dry_run: true",
            f"command:    {subcommand}",
            f"host:       {plan.host.name}",
            f"run_id:     {plan.run_id}",
            f"remote_dir: {plan.remote_dir}",
            f"log:        {plan.log_path}",
            f"profile:    {','.join(plan.profile.names)}",
            f"tmux:       {'new-session' if plan.new_session else (plan.requested_session or 'auto')}",
        ]
    )


def run_run(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    if args.dry_run:
        profile = _resolve_profile(args)
        plan = build_run_plan(
            host=host,
            local_dir=args.local_dir,
            script=args.script,
            profile=profile,
            args=tuple(args.arg),
            session=args.session,
            new_session=args.new_session,
            window=args.window,
            remote_dir=args.remote_dir,
            log_path=args.log,
            replace=args.replace,
        )
        print(_dry_run_summary(plan, subcommand="run"))
        return 0

    ensure_knuckles_master(runner=runner)
    gpu_env = _gpu_env(args, host, runner=runner)
    profile = _resolve_profile(args, extra_env=gpu_env)
    plan = build_run_plan(
        host=host,
        local_dir=args.local_dir,
        script=args.script,
        profile=profile,
        args=tuple(args.arg),
        session=args.session,
        new_session=args.new_session,
        window=args.window,
        remote_dir=args.remote_dir,
        log_path=args.log,
        replace=args.replace,
    )
    sessions = list_remote_sessions(host, runner=runner)
    decision = decide_tmux(
        sessions=sessions,
        generated_session=plan.run_id,
        requested_session=plan.requested_session,
        new_session=plan.new_session,
        window=plan.window,
    )
    upload_bundle(plan, runner=runner, popener=popener)
    launcher = write_launcher_files(plan, runner=runner)
    launch_tmux(plan, decision, launcher, runner=runner)
    write_record(_record_from_plan(plan, decision))
    print(format_summary(plan, decision))
    return 0


def run_exec(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    command = _strip_remainder(args.exec_command)
    stdin_body = sys.stdin.read() if args.stdin else None
    if args.dry_run:
        profile = _resolve_profile(args)
        plan = build_exec_plan(
            host=host,
            command=command,
            stdin_body=stdin_body,
            profile=profile,
            session=args.session,
            new_session=args.new_session,
            window=args.window,
            remote_dir=args.remote_dir,
            log_path=args.log,
        )
        print(_dry_run_summary(plan, subcommand="exec"))
        return 0

    ensure_knuckles_master(runner=runner)
    gpu_env = _gpu_env(args, host, runner=runner)
    profile = _resolve_profile(args, extra_env=gpu_env)
    plan = build_exec_plan(
        host=host,
        command=command,
        stdin_body=stdin_body,
        profile=profile,
        session=args.session,
        new_session=args.new_session,
        window=args.window,
        remote_dir=args.remote_dir,
        log_path=args.log,
    )
    sessions = list_remote_sessions(host, runner=runner)
    decision = decide_tmux(
        sessions=sessions,
        generated_session=plan.run_id,
        requested_session=plan.requested_session,
        new_session=plan.new_session,
        window=plan.window,
        require_explicit_when_not_single=True,
    )
    create_remote_dir(plan, runner=runner)
    launcher = write_launcher_files(plan, runner=runner)
    launch_tmux(plan, decision, launcher, runner=runner)
    write_record(_record_from_plan(plan, decision))
    print(format_summary(plan, decision))
    return 0


def _record_from_plan(plan, decision) -> RunRecord:
    return RunRecord(
        run_id=plan.run_id,
        kind=plan.kind,
        host=plan.host.name,
        ssh_host=plan.host.ssh_host,
        session=decision.session,
        window=decision.window,
        remote_dir=plan.remote_dir,
        log_path=plan.log_path,
        command=plan.command,
    )


def run_tail(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    record = read_record(args.run_ref)
    ensure_knuckles_master(runner=runner)
    flag = "-f" if args.follow else f"-n {int(args.lines)}"
    command = f"tail {flag} {shlex.quote(record.log_path)}"
    proc = runner(["ssh", record.ssh_host, "bash", "-lc", f"'{command}'"], text=True, shell=False)
    return int(getattr(proc, "returncode", 0))


def run_fetch(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    record = read_record(args.run_ref)
    output_dir = args.output_dir or Path("ucl-fetch") / record.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_knuckles_master(runner=runner)
    remote_dir_q = shlex.quote(record.remote_dir)
    command = (
        "set -euo pipefail; "
        f"cd {remote_dir_q}; "
        "find . -type f -regex \"./.*\\(\\.log\\|\\.json\\|\\.jsonl\\|\\.yaml\\|\\.yml\\|\\.txt\\|/.ucl_.*\\)\" "
        "-print0 | tar --null -T - -cf -"
    )
    remote = popener(["ssh", record.ssh_host, "bash", "-lc", f"'{command}'"], stdout=subprocess.PIPE)
    try:
        proc = runner(["tar", "-xf", "-", "-C", str(output_dir)], stdin=remote.stdout, capture_output=True, shell=False)
        if remote.stdout is not None:
            remote.stdout.close()
        remote_rc = remote.wait()
    finally:
        if remote.stdout is not None and not remote.stdout.closed:
            remote.stdout.close()
    if remote_rc != 0:
        raise RuntimeError(f"remote tar failed with exit {remote_rc}")
    if int(getattr(proc, "returncode", 1)) != 0:
        raise RuntimeError((getattr(proc, "stderr", b"") or b"").decode(errors="replace").strip() or "local tar failed")
    print(output_dir)
    return 0


def run_clean(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    ensure_knuckles_master(runner=runner)
    days = int(args.older_than_days)
    action = "-delete" if args.execute else "-print"
    command = f"find /tmp/ucl-machine-tools/launchers -mindepth 1 -maxdepth 1 -type d -mtime +{days} {action} 2>/dev/null || true"
    proc = runner(["ssh", host.ssh_host, "bash", "-lc", f"'{command}'"], capture_output=True, text=True, shell=False)
    if getattr(proc, "stdout", ""):
        print(proc.stdout, end="")
    return int(getattr(proc, "returncode", 0))


def _split_exec_argv(argv: list[str]) -> tuple[list[str], tuple[str, ...]]:
    if not argv or argv[0] != "exec":
        return argv, ()
    if "--" not in argv:
        return argv, ()
    idx = argv.index("--")
    return argv[:idx], tuple(argv[idx + 1 :])


def main(argv: list[str] | None = None, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parse_argv, exec_command = _split_exec_argv(raw_argv)
    args = parser.parse_args(parse_argv)
    if args.command == "exec":
        args.exec_command = exec_command
    try:
        if args.command == "status":
            return run_status(args, runner=runner)
        if args.command == "doctor":
            return run_doctor(args, runner=runner)
        if args.command == "run":
            return run_run(args, runner=runner, popener=popener)
        if args.command == "exec":
            return run_exec(args, runner=runner)
        if args.command == "tail":
            return run_tail(args, runner=runner)
        if args.command == "fetch":
            return run_fetch(args, runner=runner, popener=popener)
        if args.command == "clean":
            return run_clean(args, runner=runner)
    except Exception as exc:  # noqa: BLE001 - CLI should render concise failures.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
