"""Unified CLI for UCL machine tools."""

from __future__ import annotations

import argparse
import base64
import json
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
    parse_env,
    upload_bundle,
    write_launcher_files,
)
from ucl_machine_tools.registry import RunRecord, read_record, write_record
from ucl_machine_tools.ssh import ensure_knuckles_master

TAIL_SENTINEL_BEGIN = "UCL_TAIL_TEXT_BEGIN"
TAIL_SENTINEL_END = "UCL_TAIL_TEXT_END"
FETCH_SENTINEL_BEGIN = "UCL_FETCH_TAR_BASE64_BEGIN"
FETCH_SENTINEL_END = "UCL_FETCH_TAR_BASE64_END"
CLEAN_SENTINEL_BEGIN = "UCL_CLEAN_JSON_BEGIN"
CLEAN_SENTINEL_END = "UCL_CLEAN_JSON_END"
EXEC_SENTINEL_BEGIN = "UCL_EXEC_JSON_BEGIN"
EXEC_SENTINEL_END = "UCL_EXEC_JSON_END"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified UCL machine helper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Common use:
  ucl status 3090ti
      Show GPU availability, /tmp space, /tmp/ucl-machine-tools, and restart policy.
  ucl status recommend 3090ti --min-free-vram-gb 20
      Pick the best currently usable matching host.
  ucl doctor barbury-l
      Check one host, scratch state, and tmux sessions before work.
  ucl exec barbury-l df -h /tmp
      Run a short remote check and print output now.
  ucl exec barbury-l --cwd /tmp --timeout 60 pwd
      Run from a remote directory with a bounded timeout.
  ucl exec barbury-l --stdin < check.sh
      Run a multi-line bash snippet from stdin and print output now.
  ucl exec barbury-l --shell csh --stdin < check_torch.csh
      Run UCL/TSG csh setup snippets, such as Python/CUDA setup.
  ucl exec barbury-l --detach -- hostname
      Launch a small command in tmux and record it like a run.
  ucl run --host barbury-l --gpu auto --local-dir ./bundle --script run.sh
      Upload a local bundle and launch its script in tmux.
  ucl tail last
      Print the latest recorded run log without login noise.
  ucl fetch last
      Fetch small log/config/text artifacts from the latest recorded run.
  ucl clean barbury-l
      List old launcher dirs; add --execute only when deletion is intended.

Use 'ucl COMMAND --help' for command-specific flags.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Check GPU, scratch, and host state.")
    status.add_argument("items", nargs="*", help="optional mode and target, e.g. recommend 3090ti")
    _add_inventory_flags(status)

    doctor = subparsers.add_parser("doctor", help="Check one host, scratch, and tmux state.")
    doctor.add_argument("host")
    doctor.add_argument("--catalog", type=Path)
    doctor.add_argument("--timeout-seconds", type=int, default=8)

    run = subparsers.add_parser("run", help="Upload a local bundle and launch it in tmux.")
    run.add_argument("--host", required=True)
    run.add_argument("--local-dir", required=True, type=Path)
    run.add_argument("--script", required=True)
    _add_launch_common_flags(run)
    run.add_argument("--arg", action="append", default=[], help="script argument; repeat for multiple args")
    run.add_argument("--replace", action="store_true", help="replace an existing non-empty remote bundle dir")

    exec_parser = subparsers.add_parser("exec", help="Run a small remote command or snippet.")
    _configure_exec_parser(exec_parser)

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
    parser.add_argument("--env", action="append", default=[], help="remote env KEY=VALUE; repeat for multiple vars")
    parser.add_argument("--gpu", help="GPU id or auto")
    parser.add_argument("--shell", choices=("bash", "csh"), default="bash", help="shell used for the generated payload")
    parser.add_argument("--session", help="tmux session to use or create")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session")
    parser.add_argument("--window", help="tmux window name")
    parser.add_argument("--remote-dir", help="remote bundle dir under /tmp/ucl-machine-tools/launchers")
    parser.add_argument("--log", help="remote log path; default is <remote-dir>/run.log")
    parser.add_argument("--dry-run", action="store_true")


def _configure_exec_parser(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.description = "Run a small remote command/snippet; synchronous by default."
    parser.epilog = """\
Examples:
  ucl exec barbury-l df -h /tmp
  ucl exec barbury-l -- python3 -c 'print("hi")'
  ucl exec barbury-l --cwd /tmp --timeout 60 pwd
  ucl exec barbury-l --stdin < check.sh
  ucl exec barbury-l --shell csh --stdin < check_torch.csh
  ucl exec barbury-l --detach --new-session -- hostname
"""
    parser.add_argument("host")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--env", action="append", default=[], help="remote env KEY=VALUE; repeat for multiple vars")
    parser.add_argument("--gpu", help="GPU id or auto")
    parser.add_argument("--shell", choices=("bash", "csh"), default="bash", help="shell used only with --stdin")
    parser.add_argument("--stdin", action="store_true", help="read a script from stdin")
    parser.add_argument("--timeout", type=float, default=60.0, help="remote command timeout in seconds; 0 disables")
    parser.add_argument("--cwd", help="remote working directory for synchronous exec")
    parser.add_argument("--json", action="store_true", help="emit returncode/stdout/stderr as JSON")
    parser.add_argument("--detach", action="store_true", help="launch in tmux and record a run instead of printing output now")
    parser.add_argument("--session", help="tmux session to use or create; requires --detach")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session; requires --detach")
    parser.add_argument("--window", help="tmux window name; requires --detach")
    parser.add_argument("--remote-dir", help="remote bundle dir under /tmp/ucl-machine-tools/launchers; requires --detach")
    parser.add_argument("--log", help="remote log path; requires --detach")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("exec_command", nargs=argparse.REMAINDER, metavar="COMMAND")


def _build_exec_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ucl exec")
    _configure_exec_parser(parser)
    return parser


def _parse_exec_argv(tokens: list[str]) -> argparse.Namespace:
    parser = _build_exec_parser()
    if not tokens or tokens[0] in {"-h", "--help"}:
        return parser.parse_args(tokens)
    host = tokens[0]
    if host.startswith("-"):
        parser.error("HOST is required before exec options")
    values: dict[str, Any] = {
        "command": "exec",
        "host": host,
        "catalog": None,
        "env": [],
        "gpu": None,
        "shell": "bash",
        "stdin": False,
        "timeout": 60.0,
        "cwd": None,
        "json": False,
        "detach": False,
        "session": None,
        "new_session": False,
        "window": None,
        "remote_dir": None,
        "log": None,
        "dry_run": False,
        "exec_command": (),
    }
    no_value_flags = {
        "--stdin": "stdin",
        "--detach": "detach",
        "--new-session": "new_session",
        "--json": "json",
        "--dry-run": "dry_run",
    }
    value_flags = {
        "--catalog": "catalog",
        "--env": "env",
        "--gpu": "gpu",
        "--shell": "shell",
        "--timeout": "timeout",
        "--cwd": "cwd",
        "--session": "session",
        "--window": "window",
        "--remote-dir": "remote_dir",
        "--log": "log",
    }
    rest = tokens[1:]
    command: list[str] = []
    idx = 0
    while idx < len(rest):
        token = rest[idx]
        if token == "--":
            command = rest[idx + 1 :]
            break
        if token in {"-h", "--help"}:
            return parser.parse_args(["--help"])
        if token in no_value_flags:
            values[no_value_flags[token]] = True
            idx += 1
            continue
        if token in value_flags:
            if idx + 1 >= len(rest):
                parser.error(f"{token} requires a value")
            raw_value = rest[idx + 1]
            key = value_flags[token]
            if key == "env":
                values["env"].append(raw_value)
            elif key == "catalog":
                values[key] = Path(raw_value)
            elif key == "timeout":
                try:
                    values[key] = float(raw_value)
                except ValueError:
                    parser.error(f"{token} must be a number")
            elif key == "shell":
                if raw_value not in {"bash", "csh"}:
                    parser.error("--shell must be one of: bash, csh")
                values[key] = raw_value
            else:
                values[key] = raw_value
            idx += 2
            continue
        if token.startswith("-"):
            parser.error(f"unknown ucl exec option: {token}; use '--' before remote commands that start with '-'")
        command = rest[idx:]
        break

    values["exec_command"] = tuple(command)
    if values["timeout"] < 0:
        parser.error("--timeout must be >= 0")
    if values["stdin"] and command:
        parser.error("--stdin cannot be used with COMMAND arguments")
    if not values["stdin"] and not command:
        parser.error("no remote command provided; use 'ucl exec HOST COMMAND...' or 'ucl exec HOST --stdin < script.sh'")
    tmux_only = []
    for flag, key in (
        ("--session", "session"),
        ("--new-session", "new_session"),
        ("--window", "window"),
        ("--remote-dir", "remote_dir"),
        ("--log", "log"),
    ):
        if values[key]:
            tmux_only.append(flag)
    if not values["detach"] and tmux_only:
        parser.error(f"{', '.join(tmux_only)} require --detach")
    if values["detach"]:
        sync_only = []
        if values["cwd"] is not None:
            sync_only.append("--cwd")
        if values["json"]:
            sync_only.append("--json")
        if values["timeout"] != 60.0:
            sync_only.append("--timeout")
        if sync_only:
            parser.error(f"{', '.join(sync_only)} are only supported for synchronous exec")
    return argparse.Namespace(**values)


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


def _resolve_env(args: argparse.Namespace, host: HostSpec, *, runner) -> tuple[tuple[str, str], ...]:
    return (*parse_env(args.env), *_gpu_env(args, host, runner=runner))


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


def run_doctor(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    ensure_knuckles_master(runner=runner)
    row = inventory.collect([host], runner=runner, jobs=1, timeout_seconds=args.timeout_seconds)[0]
    sessions = list_remote_sessions(host, runner=runner)
    print(f"host:          {host.name}")
    print(f"status:        {row.get('status')}")
    print(f"tmp_scratch:  {'yes' if (row.get('scratch') or {}).get('exists') else 'no'}")
    print(f"tmux_sessions: {', '.join(sessions) if sessions else 'none'}")
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
            f"shell:      {plan.shell}",
            f"tmux:       {'new-session' if plan.new_session else (plan.requested_session or 'auto')}",
        ]
    )


def _sync_exec_dry_run_summary(args: argparse.Namespace, host: HostSpec, command: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "dry_run: true",
            "command:    exec",
            "mode:       sync",
            f"host:       {host.name}",
            f"shell:      {args.shell}",
            f"cwd:        {args.cwd or '-'}",
            f"timeout:    {'none' if args.timeout == 0 else args.timeout:g}",
            f"stdin:      {'yes' if args.stdin else 'no'}",
            f"argv:       {json.dumps(list(command))}",
        ]
    )


def _sync_exec_source(params: dict[str, Any]) -> str:
    params_json = json.dumps(params, sort_keys=True)
    return f"""
import base64
import json
import os
import subprocess
import sys
BEGIN={json.dumps(EXEC_SENTINEL_BEGIN)}
END={json.dumps(EXEC_SENTINEL_END)}
PARAMS=json.loads({params_json!r})

env = os.environ.copy()
env.update(PARAMS.get("env", {{}}))
cwd = PARAMS.get("cwd") or None
timeout_value = PARAMS.get("timeout")
timeout = None if timeout_value in (None, 0) else float(timeout_value)
stdout = b""
stderr = b""
returncode = 0
timed_out = False
wrapper_error = False

try:
    if PARAMS["mode"] == "stdin":
        argv = ["csh", "-f"] if PARAMS.get("shell") == "csh" else ["bash"]
        proc = subprocess.run(
            argv,
            input=base64.b64decode(PARAMS.get("stdin_b64", "")),
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
    else:
        proc = subprocess.run(
            PARAMS["argv"],
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
    stdout = proc.stdout or b""
    stderr = proc.stderr or b""
    returncode = int(proc.returncode)
except subprocess.TimeoutExpired as exc:
    timed_out = True
    returncode = 124
    stdout = exc.stdout or b""
    stderr = exc.stderr or b""
    message = f"ucl exec timed out after {{timeout_value}} seconds\\n".encode()
    stderr = stderr + message
except Exception as exc:
    wrapper_error = True
    returncode = 127
    stderr = f"{{type(exc).__name__}}: {{exc}}\\n".encode()

payload = {{
    "schema_version": 1,
    "returncode": returncode,
    "stdout_b64": base64.b64encode(stdout).decode("ascii"),
    "stderr_b64": base64.b64encode(stderr).decode("ascii"),
    "timed_out": timed_out,
    "wrapper_error": wrapper_error,
}}
print(BEGIN)
print(json.dumps(payload, sort_keys=True))
print(END)
"""


def _parse_sync_exec_result(stdout: str) -> dict[str, Any]:
    payload = json.loads(_extract_between(stdout, EXEC_SENTINEL_BEGIN, EXEC_SENTINEL_END, label="exec"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported exec result schema: {payload.get('schema_version')!r}")
    return {
        "returncode": int(payload.get("returncode", 127)),
        "stdout": base64.b64decode(payload.get("stdout_b64", "")),
        "stderr": base64.b64decode(payload.get("stderr_b64", "")),
        "timed_out": bool(payload.get("timed_out", False)),
        "wrapper_error": bool(payload.get("wrapper_error", False)),
    }


def _decode_stream(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _format_sync_exec_json(
    *,
    host: HostSpec,
    command: tuple[str, ...],
    args: argparse.Namespace,
    result: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "host": host.name,
            "ssh_host": host.ssh_host,
            "command": list(command),
            "cwd": args.cwd,
            "timeout": None if args.timeout == 0 else args.timeout,
            "timed_out": result["timed_out"],
            "returncode": result["returncode"],
            "stdout": _decode_stream(result["stdout"]),
            "stderr": _decode_stream(result["stderr"]),
        },
        indent=2,
        sort_keys=True,
    )


def run_exec_sync(
    args: argparse.Namespace,
    *,
    host: HostSpec,
    command: tuple[str, ...],
    stdin_body: str | None,
    runner=subprocess.run,
) -> int:
    if args.dry_run:
        print(_sync_exec_dry_run_summary(args, host, command))
        return 0
    ensure_knuckles_master(runner=runner)
    env = _resolve_env(args, host, runner=runner)
    params = {
        "mode": "stdin" if args.stdin else "command",
        "argv": list(command),
        "stdin_b64": base64.b64encode((stdin_body or "").encode("utf-8")).decode("ascii"),
        "env": dict(env),
        "shell": args.shell,
        "cwd": args.cwd,
        "timeout": args.timeout,
    }
    proc = runner(
        _ssh_python_argv(host.ssh_host),
        input=_sync_exec_source(params),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = _strip_remote_noise((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip())
        raise RuntimeError(detail or "remote exec wrapper failed")
    result = _parse_sync_exec_result(getattr(proc, "stdout", "") or "")
    if args.json:
        print(_format_sync_exec_json(host=host, command=command, args=args, result=result))
    else:
        if result["stdout"]:
            sys.stdout.write(_decode_stream(result["stdout"]))
        if result["stderr"]:
            sys.stderr.write(_decode_stream(result["stderr"]))
    return int(result["returncode"])


def run_run(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    if args.dry_run:
        plan = build_run_plan(
            host=host,
            local_dir=args.local_dir,
            script=args.script,
            args=tuple(args.arg),
            env=parse_env(args.env),
            shell=args.shell,
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
    env = _resolve_env(args, host, runner=runner)
    plan = build_run_plan(
        host=host,
        local_dir=args.local_dir,
        script=args.script,
        args=tuple(args.arg),
        env=env,
        shell=args.shell,
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
    if not args.detach:
        return run_exec_sync(args, host=host, command=command, stdin_body=stdin_body, runner=runner)
    if args.dry_run:
        plan = build_exec_plan(
            host=host,
            command=command,
            stdin_body=stdin_body,
            env=parse_env(args.env),
            shell=args.shell,
            session=args.session,
            new_session=args.new_session,
            window=args.window,
            remote_dir=args.remote_dir,
            log_path=args.log,
        )
        print(_dry_run_summary(plan, subcommand="exec"))
        return 0

    ensure_knuckles_master(runner=runner)
    env = _resolve_env(args, host, runner=runner)
    plan = build_exec_plan(
        host=host,
        command=command,
        stdin_body=stdin_body,
        env=env,
        shell=args.shell,
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


def _tail_source(path: str, lines: int) -> str:
    return f"""
import collections
import json
import sys
BEGIN={json.dumps(TAIL_SENTINEL_BEGIN)}
END={json.dumps(TAIL_SENTINEL_END)}
PATH={json.dumps(path)}
LINES={int(lines)}
print(BEGIN)
with open(PATH, "r", encoding="utf-8", errors="replace") as handle:
    for line in collections.deque(handle, maxlen=LINES):
        sys.stdout.write(line)
print(END)
"""


def _extract_between(stdout: str, begin: str, end: str, *, label: str) -> str:
    lines = stdout.splitlines(keepends=True)
    begin_idx = next((idx for idx, line in enumerate(lines) if line.strip() == begin), None)
    if begin_idx is None:
        raise RuntimeError(f"{label} sentinel not found")
    end_idx = next((idx for idx in range(begin_idx + 1, len(lines)) if lines[idx].strip() == end), None)
    if end_idx is None:
        raise RuntimeError(f"{label} sentinel end not found")
    return "".join(lines[begin_idx + 1 : end_idx])


def _extract_tail(stdout: str) -> str:
    return _extract_between(stdout, TAIL_SENTINEL_BEGIN, TAIL_SENTINEL_END, label="tail")


def _tail_follow_source(path: str, lines: int) -> str:
    return f"""
import json
import subprocess
BEGIN={json.dumps(TAIL_SENTINEL_BEGIN)}
PATH={json.dumps(path)}
LINES={int(lines)}
print(BEGIN, flush=True)
raise SystemExit(subprocess.call(["tail", "-n", str(LINES), "-f", PATH]))
"""


def _strip_remote_noise(text: str) -> str:
    noisy_markers = ("VBoxManage", "VirtualBox")
    return "".join(line for line in text.splitlines(keepends=True) if not any(marker in line for marker in noisy_markers))


def _ssh_python_argv(host: str) -> list[str]:
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", host, "python3", "-"]


def run_tail(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    record = read_record(args.run_ref)
    ensure_knuckles_master(runner=runner)
    if args.follow:
        proc = popener(
            _ssh_python_argv(record.ssh_host),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("failed to open remote tail pipes")
        proc.stdin.write(_tail_follow_source(record.log_path, int(args.lines)))
        proc.stdin.close()
        started = False
        try:
            while True:
                line = proc.stdout.readline()
                if line == "":
                    break
                if not started:
                    if line.strip() == TAIL_SENTINEL_BEGIN:
                        started = True
                    continue
                print(line, end="", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            return 130
        returncode = int(proc.wait())
        stderr = ""
        if proc.stderr is not None:
            stderr = _strip_remote_noise(proc.stderr.read())
        if returncode != 0 and stderr:
            print(stderr, file=sys.stderr, end="")
        return returncode
    proc = runner(
        _ssh_python_argv(record.ssh_host),
        input=_tail_source(record.log_path, int(args.lines)),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        raise RuntimeError(detail or "remote tail failed")
    print(_extract_tail(getattr(proc, "stdout", "") or ""), end="")
    return int(getattr(proc, "returncode", 0))


def run_fetch(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    record = read_record(args.run_ref)
    output_dir = args.output_dir or Path("ucl-fetch") / record.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_knuckles_master(runner=runner)
    proc = runner(
        _ssh_python_argv(record.ssh_host),
        input=_fetch_source(record.remote_dir),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = _strip_remote_noise((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip())
        raise RuntimeError(detail or "remote fetch failed")
    tar_data = base64.b64decode(_extract_between(getattr(proc, "stdout", "") or "", FETCH_SENTINEL_BEGIN, FETCH_SENTINEL_END, label="fetch"))
    local = runner(["tar", "-xf", "-", "-C", str(output_dir)], input=tar_data, capture_output=True, shell=False)
    if int(getattr(local, "returncode", 1)) != 0:
        raise RuntimeError((getattr(local, "stderr", b"") or b"").decode(errors="replace").strip() or "local tar failed")
    print(output_dir)
    return 0


def _fetch_source(remote_dir: str) -> str:
    return f"""
import base64
import io
import json
import os
import tarfile
import sys
BEGIN={json.dumps(FETCH_SENTINEL_BEGIN)}
END={json.dumps(FETCH_SENTINEL_END)}
REMOTE_DIR={json.dumps(remote_dir)}
SUFFIXES=(".log", ".json", ".jsonl", ".yaml", ".yml", ".txt")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tar:
    if os.path.isdir(REMOTE_DIR):
        for root, _, files in os.walk(REMOTE_DIR):
            for name in sorted(files):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, REMOTE_DIR)
                base = os.path.basename(rel)
                if rel.endswith(SUFFIXES) or base.startswith(".ucl_"):
                    tar.add(path, arcname=rel, recursive=False)
print(BEGIN)
sys.stdout.write(base64.b64encode(buf.getvalue()).decode("ascii"))
sys.stdout.write("\\n")
print(END)
"""


def run_clean(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    ensure_knuckles_master(runner=runner)
    proc = runner(
        _ssh_python_argv(host.ssh_host),
        input=_clean_source(int(args.older_than_days), bool(args.execute)),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = _strip_remote_noise((getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip())
        raise RuntimeError(detail or "remote clean failed")
    payload = json.loads(_extract_between(getattr(proc, "stdout", "") or "", CLEAN_SENTINEL_BEGIN, CLEAN_SENTINEL_END, label="clean"))
    for path in payload.get("paths", []):
        print(path)
    return int(getattr(proc, "returncode", 0))


def _clean_source(days: int, execute: bool) -> str:
    return f"""
import json
import os
import shutil
import time
BEGIN={json.dumps(CLEAN_SENTINEL_BEGIN)}
END={json.dumps(CLEAN_SENTINEL_END)}
ROOT="/tmp/ucl-machine-tools/launchers"
DAYS={int(days)}
EXECUTE={bool(execute)!r}
cutoff = time.time() - DAYS * 86400
paths = []
if os.path.isdir(ROOT):
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if os.path.isdir(path) and stat.st_mtime < cutoff:
            paths.append(path)
            if EXECUTE:
                shutil.rmtree(path)
print(BEGIN)
print(json.dumps({{"schema_version": 1, "paths": paths}}, sort_keys=True))
print(END)
"""


def main(argv: list[str] | None = None, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "exec":
        try:
            args = _parse_exec_argv(raw_argv[1:])
        except SystemExit as exc:
            return int(exc.code or 0)
        try:
            return run_exec(args, runner=runner)
        except Exception as exc:  # noqa: BLE001 - CLI should render concise failures.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        if args.command == "status":
            return run_status(args, runner=runner)
        if args.command == "doctor":
            return run_doctor(args, runner=runner)
        if args.command == "run":
            return run_run(args, runner=runner, popener=popener)
        if args.command == "tail":
            return run_tail(args, runner=runner, popener=popener)
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
