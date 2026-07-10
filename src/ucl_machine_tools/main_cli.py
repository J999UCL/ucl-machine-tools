"""Unified CLI for UCL machine tools."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import shlex
from pathlib import Path
from typing import Any

from ucl_machine_tools.hosts import HostSpec, load_catalog, parse_selector
from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import envcheck
from ucl_machine_tools import inventory
from ucl_machine_tools.launch import (
    build_exec_plan,
    build_run_plan,
    create_remote_dir,
    decide_tmux,
    default_remote_root,
    format_summary,
    launch_tmux,
    list_remote_sessions,
    parse_env,
    upload_bundle,
    write_launcher_files,
)
from ucl_machine_tools.registry import RunRecord, list_records, read_record, write_record
from ucl_machine_tools.ssh import build_remote_python_argv, ensure_knuckles_master

TAIL_SENTINEL_BEGIN = "UCL_TAIL_TEXT_BEGIN"
TAIL_SENTINEL_END = "UCL_TAIL_TEXT_END"
FETCH_SENTINEL_BEGIN = "UCL_FETCH_TAR_BASE64_BEGIN"
FETCH_SENTINEL_END = "UCL_FETCH_TAR_BASE64_END"
CLEAN_SENTINEL_BEGIN = "UCL_CLEAN_JSON_BEGIN"
CLEAN_SENTINEL_END = "UCL_CLEAN_JSON_END"
EXEC_SENTINEL_BEGIN = "UCL_EXEC_JSON_BEGIN"
EXEC_SENTINEL_END = "UCL_EXEC_JSON_END"
ERROR_SNIPPET_CHARS = 800
DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified UCL machine helper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Common workflows:
  Inspect machines:
    ucl status 3090ti
    ucl status barbury-l canada-l
    ucl status recommend 3090ti --min-free-vram-gb 20
    ucl doctor barbury-l

  Run quick synchronous commands:
    ucl exec barbury-l df -h /tmp
    ucl exec barbury-l canada-l -- hostname
    ucl exec barbury-l --cwd /tmp --timeout 60 --connect-timeout 30 pwd
    ucl exec 3090ti --gpu auto --min-free-vram-gb 20 -- nvidia-smi
    ucl exec barbury-l --stdin < check.sh
    ucl exec barbury-l --shell csh --stdin < check_torch.csh

  Launch and manage tmux-backed jobs:
    ucl exec barbury-l --detach --new-session -- hostname
    ucl run --host barbury-l --new-session --gpu auto --min-free-vram-gb 20 --local-dir ./bundle --script run.sh
    ucl jobs
    ucl info last
    ucl tail last
    ucl fetch last
    ucl stop RUN_ID
    ucl clean barbury-l

  Copy data:
    ucl copy /tmp/a barbury-l:/tmp/a --verify size
    ucl copy barbury-l:/tmp/a barnacle-l:/tmp/a -- --partial --info=progress2 --exclude '*.pt'

  Check a remote scratch root:
    ucl env barbury-l --remote-root /tmp/ucl-machine-tools/fpt --json

Use 'ucl COMMAND --help' for command-specific flags.
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Check GPU, scratch, and host state.")
    status.add_argument("items", nargs="*", help="optional mode and one or more targets, e.g. recommend 3090ti timeshare")
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
    clean.add_argument("--remote-root", help="remote launcher root; defaults to UCL_LAUNCH_ROOT or /tmp/ucl-machine-tools/launchers")
    clean.add_argument("--execute", action="store_true", help="delete listed launcher directories")
    clean.add_argument("--older-than-days", type=int, default=7)

    jobs = subparsers.add_parser("jobs", help="List recorded tmux-backed jobs.")
    jobs.add_argument("--json", action="store_true")
    jobs.add_argument("--all", action="store_true")
    jobs.add_argument("--catalog", type=Path)
    jobs.add_argument("--timeout-seconds", type=int, default=8)

    info = subparsers.add_parser("info", help="Show one recorded job.")
    info.add_argument("run_ref", nargs="?", default="last")
    info.add_argument("--json", action="store_true")
    info.add_argument("--catalog", type=Path)
    info.add_argument("--timeout-seconds", type=int, default=8)

    stop = subparsers.add_parser("stop", help="Stop one recorded tmux job.")
    stop.add_argument("run_ref", help="run id to stop; use 'last --yes' only when intentional")
    stop.add_argument("--signal", choices=("TERM", "KILL"), default="TERM")
    stop.add_argument("--yes", action="store_true", help="allow stopping the latest recorded run via 'last'")
    stop.add_argument("--json", action="store_true")
    stop.add_argument("--timeout-seconds", type=int, default=8)

    copy = subparsers.add_parser("copy", help="Copy local/remote paths with rsync.")
    copy.add_argument("src")
    copy.add_argument("dst")
    copy.add_argument("--catalog", type=Path)
    copy.add_argument("--verify", choices=("size", "sha256", "none"), default="none")
    copy.add_argument("--partial", action="store_true")
    copy.add_argument("--progress", action="store_true")
    copy.add_argument("--dry-run", action="store_true")
    copy.add_argument("--json", action="store_true")

    env = subparsers.add_parser("env", help="Check one host and a remote scratch root.")
    env.add_argument("host")
    env.add_argument("--catalog", type=Path)
    env.add_argument("--remote-root", required=True)
    env.add_argument("--gpu", help="GPU id or auto")
    env.add_argument("--min-free-vram-gb", type=float, default=DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB)
    env.add_argument("--create", action="store_true")
    env.add_argument("--json", action="store_true")

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
    parser.add_argument("--min-free-vram-gb", type=float, default=DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB, help="minimum free VRAM for --gpu auto")
    parser.add_argument("--project", help="project tag stored in run provenance")
    parser.add_argument("--shell", choices=("bash", "csh"), default="bash", help="shell used for the generated payload")
    parser.add_argument("--session", help="tmux session to use or create")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session")
    parser.add_argument("--window", help="tmux window name")
    parser.add_argument("--remote-dir", help="exact remote bundle dir; must be under the selected launcher root")
    parser.add_argument("--remote-root", help="remote launcher root; defaults to UCL_LAUNCH_ROOT or /tmp/ucl-machine-tools/launchers")
    parser.add_argument("--log", help="remote log path; default is <remote-dir>/run.log")
    parser.add_argument("--dry-run", action="store_true")


def _configure_exec_parser(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.description = "Run a small remote command/snippet; synchronous by default."
    parser.epilog = """\
Examples:
  ucl exec barbury-l df -h /tmp
  ucl exec barbury-l canada-l -- hostname
  ucl exec barbury-l canada-l barnacle-l -- hostname
  ucl exec 3090ti -- df -h /tmp
  ucl exec barbury-l -- python3 -c 'print("hi")'
  ucl exec barbury-l --cwd /tmp --timeout 60 --connect-timeout 30 pwd
  ucl exec barbury-l --stdin < check.sh
  ucl exec barbury-l --shell csh --stdin < check_torch.csh
  ucl exec barbury-l --detach --new-session -- hostname
"""
    parser.add_argument("host", metavar="HOST_OR_SELECTOR", help="first host/selector; add more before -- for multi-host sync exec")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--env", action="append", default=[], help="remote env KEY=VALUE; repeat for multiple vars")
    parser.add_argument("--gpu", help="GPU id or auto")
    parser.add_argument("--min-free-vram-gb", type=float, default=DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB, help="minimum free VRAM for --gpu auto")
    parser.add_argument("--project", help="project tag stored in run provenance")
    parser.add_argument("--shell", choices=("bash", "csh"), default="bash", help="shell used only with --stdin")
    parser.add_argument("--stdin", action="store_true", help="read a script from stdin")
    parser.add_argument("--timeout", type=float, default=60.0, help="remote command timeout in seconds; 0 disables")
    parser.add_argument("--connect-timeout", type=int, default=30, help="SSH connect timeout in seconds; 0 disables")
    parser.add_argument("--cwd", help="remote working directory for synchronous exec")
    parser.add_argument("--json", action="store_true", help="emit returncode/stdout/stderr as JSON")
    parser.add_argument("--detach", action="store_true", help="launch in tmux and record a run instead of printing output now")
    parser.add_argument("--session", help="tmux session to use or create; requires --detach")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session; requires --detach")
    parser.add_argument("--window", help="tmux window name; requires --detach")
    parser.add_argument("--remote-dir", help="remote bundle dir under /tmp/ucl-machine-tools/launchers; requires --detach")
    parser.add_argument("--log", help="remote log path; requires --detach")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("exec_command", nargs=argparse.REMAINDER, metavar="[HOST_OR_SELECTOR ...] -- COMMAND")


def _build_exec_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ucl exec")
    _configure_exec_parser(parser)
    return parser


def _exec_catalog_path_from_tokens(tokens: list[str]) -> Path | None:
    for idx, token in enumerate(tokens):
        if token == "--":
            break
        if token == "--catalog" and idx + 1 < len(tokens):
            return Path(tokens[idx + 1])
    return None


def _is_exec_target_token(token: str, *, catalog_path: Path | None) -> bool:
    if token.startswith("-"):
        return False
    try:
        parse_selector(token, catalog=load_catalog(catalog_path))
    except Exception:  # noqa: BLE001 - parser uses this only as a yes/no discriminator.
        return False
    return True


def _parse_exec_argv(tokens: list[str]) -> argparse.Namespace:
    parser = _build_exec_parser()
    if not tokens or tokens[0] in {"-h", "--help"}:
        return parser.parse_args(tokens)
    catalog_path_hint = _exec_catalog_path_from_tokens(tokens)
    first_host = tokens[0]
    if first_host.startswith("-"):
        parser.error("HOST is required before exec options")
    hosts = [first_host]
    values: dict[str, Any] = {
        "command": "exec",
        "host": first_host,
        "hosts": tuple(hosts),
        "catalog": None,
        "env": [],
        "gpu": None,
        "min_free_vram_gb": DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB,
        "project": None,
        "shell": "bash",
        "stdin": False,
        "timeout": 60.0,
        "connect_timeout": 30,
        "cwd": None,
        "json": False,
        "detach": False,
        "session": None,
        "new_session": False,
        "window": None,
        "remote_dir": None,
        "remote_root": None,
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
        "--min-free-vram-gb": "min_free_vram_gb",
        "--project": "project",
        "--shell": "shell",
        "--timeout": "timeout",
        "--connect-timeout": "connect_timeout",
        "--cwd": "cwd",
        "--session": "session",
        "--window": "window",
        "--remote-dir": "remote_dir",
        "--remote-root": "remote_root",
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
            elif key == "connect_timeout":
                try:
                    values[key] = int(raw_value)
                except ValueError:
                    parser.error(f"{token} must be an integer")
            elif key == "min_free_vram_gb":
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
        if "--" in rest[idx + 1 :] and _is_exec_target_token(token, catalog_path=catalog_path_hint):
            hosts.append(token)
            idx += 1
            continue
        command = rest[idx:]
        break

    values["hosts"] = tuple(hosts)
    values["exec_command"] = tuple(command)
    if values["timeout"] < 0:
        parser.error("--timeout must be >= 0")
    if values["connect_timeout"] < 0:
        parser.error("--connect-timeout must be >= 0")
    if values["min_free_vram_gb"] < 0:
        parser.error("--min-free-vram-gb must be >= 0")
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
        ("--remote-root", "remote_root"),
        ("--log", "log"),
    ):
        if values[key]:
            tmux_only.append(flag)
    if not values["detach"] and tmux_only:
        parser.error(f"{', '.join(tmux_only)} require --detach")
    if values["detach"]:
        if len(values["hosts"]) != 1:
            parser.error("multi-host exec is synchronous only; remove --detach or run separate detached jobs")
        sync_only = []
        if values["cwd"] is not None:
            sync_only.append("--cwd")
        if values["json"]:
            sync_only.append("--json")
        if values["timeout"] != 60.0:
            sync_only.append("--timeout")
        if values["connect_timeout"] != 30:
            sync_only.append("--connect-timeout")
        if sync_only:
            parser.error(f"{', '.join(sync_only)} are only supported for synchronous exec")
    return argparse.Namespace(**values)


def _build_copy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ucl copy",
        description="Copy local/remote paths with rsync.",
        epilog="Use '--' before raw rsync args, e.g. ucl copy SRC DST -- --exclude '*.pt'.",
    )
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--verify", choices=("size", "sha256", "none"), default="none")
    parser.add_argument("--partial", action="store_true", help="add rsync --partial")
    parser.add_argument("--progress", action="store_true", help="add rsync --info=progress2")
    parser.add_argument("--dry-run", action="store_true", help="print/perform rsync dry-run")
    parser.add_argument("--json", action="store_true")
    return parser


def _parse_copy_argv(tokens: list[str]) -> argparse.Namespace:
    parser = _build_copy_parser()
    if not tokens or tokens[0] in {"-h", "--help"}:
        return parser.parse_args(tokens)
    values: dict[str, Any] = {
        "command": "copy",
        "src": None,
        "dst": None,
        "catalog": None,
        "verify": "none",
        "partial": False,
        "progress": False,
        "dry_run": False,
        "json": False,
        "rsync_args": (),
    }
    operands: list[str] = []
    raw_rsync_args: list[str] = []
    no_value_flags = {
        "--partial": "partial",
        "--progress": "progress",
        "--dry-run": "dry_run",
        "--json": "json",
    }
    value_flags = {
        "--catalog": "catalog",
        "--verify": "verify",
    }
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            raw_rsync_args = tokens[idx + 1 :]
            break
        if token in {"-h", "--help"}:
            return parser.parse_args(["--help"])
        if token in no_value_flags:
            values[no_value_flags[token]] = True
            idx += 1
            continue
        if token in value_flags:
            if idx + 1 >= len(tokens):
                parser.error(f"{token} requires a value")
            raw_value = tokens[idx + 1]
            key = value_flags[token]
            if key == "catalog":
                values[key] = Path(raw_value)
            elif key == "verify":
                if raw_value not in {"size", "sha256", "none"}:
                    parser.error("--verify must be one of: size, sha256, none")
                values[key] = raw_value
            idx += 2
            continue
        if token.startswith("-"):
            parser.error(f"unknown ucl copy option: {token}; use '--' before raw rsync options")
        operands.append(token)
        idx += 1

    if len(operands) != 2:
        parser.error("copy requires exactly SRC and DST")
    values["src"] = operands[0]
    values["dst"] = operands[1]
    values["rsync_args"] = tuple(raw_rsync_args)
    return argparse.Namespace(**values)


def _resolve_one_host(selector: str, *, catalog_path: Path | None) -> HostSpec:
    catalog = load_catalog(catalog_path)
    hosts = parse_selector(selector, catalog=catalog)
    if len(hosts) != 1:
        raise ValueError(f"selector must resolve to exactly one host, got {len(hosts)} for {selector!r}")
    return hosts[0]


def _resolve_exec_hosts(args: argparse.Namespace) -> list[HostSpec]:
    catalog = load_catalog(args.catalog)
    return _resolve_status_targets(tuple(getattr(args, "hosts", (args.host,))), catalog=catalog)


def _status_mode_and_targets(items: list[str], selector: str | None) -> tuple[str, tuple[str, ...]]:
    if selector:
        return "check", (selector,)
    if not items:
        return "check", ("all",)
    if items[0] in {"check", "gpus", "state", "recommend"}:
        return items[0], tuple(items[1:] or ("all",))
    return "check", tuple(items)


def _resolve_status_targets(targets: tuple[str, ...], *, catalog: dict[str, HostSpec]) -> list[HostSpec]:
    return parse_selector(",".join(targets), catalog=catalog)


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
) -> tuple[tuple[str, str], ...]:
    if not getattr(args, "gpu", None):
        return ()
    if args.gpu != "auto":
        return (("CUDA_VISIBLE_DEVICES", str(args.gpu)),)
    min_free_vram_gb = float(getattr(args, "min_free_vram_gb", DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB))
    if min_free_vram_gb < 0:
        raise ValueError("--min-free-vram-gb must be >= 0")
    rows = inventory.collect([host], runner=runner, jobs=1, min_free_vram_gb=min_free_vram_gb)
    gpu_id = _best_free_gpu(rows[0], min_free_vram_gb=min_free_vram_gb)
    return (("CUDA_VISIBLE_DEVICES", gpu_id),)


def _resolve_env(args: argparse.Namespace, host: HostSpec, *, runner) -> tuple[tuple[str, str], ...]:
    return (*parse_env(args.env), *_gpu_env(args, host, runner=runner))


def _selected_gpu(env: tuple[tuple[str, str], ...]) -> str:
    for key, value in env:
        if key == "CUDA_VISIBLE_DEVICES":
            return value
    return ""


def _git_sha(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            shell=False,
        )
    except Exception:
        return ""
    if int(getattr(proc, "returncode", 1)) != 0:
        return ""
    return (getattr(proc, "stdout", "") or "").strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _script_path_for_plan(plan) -> Path | None:
    if plan.local_dir is None or not plan.command:
        return None
    if plan.command[0] != "bash" or len(plan.command) < 2:
        return None
    return (plan.local_dir / plan.command[1]).resolve()


def _provenance_for_plan(
    plan,
    *,
    args: argparse.Namespace,
    env: tuple[tuple[str, str], ...],
    stdin_body: str | None,
) -> dict[str, Any]:
    script_path = _script_path_for_plan(plan)
    script_sha = ""
    if script_path is not None and script_path.is_file():
        script_sha = _file_sha256(script_path)
    elif stdin_body is not None:
        script_sha = _sha256_bytes(stdin_body.encode("utf-8"))
    bundle_path = str(plan.local_dir) if plan.local_dir is not None else ""
    git_path = plan.local_dir or Path.cwd()
    return {
        "project": getattr(args, "project", None) or "",
        "local_git_sha": _git_sha(git_path),
        "script_sha256": script_sha,
        "bundle_path": bundle_path,
        "selected_gpu": _selected_gpu(env),
        "env": {key: "<redacted>" for key, _ in sorted(env)},
        "env_keys": sorted({key for key, _ in env}),
        "remote_root": plan.remote_root,
    }


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
    mode, targets = _status_mode_and_targets(args.items, args.selector)
    catalog = load_catalog(args.catalog)
    selected = _resolve_status_targets(targets, catalog=catalog)
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
    sessions = list_remote_sessions(host, runner=runner, timeout_seconds=args.timeout_seconds)
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
            f"remote_root: {plan.remote_root}",
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
            f"connect:    {'none' if args.connect_timeout == 0 else args.connect_timeout}",
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


def _compact_text(text: str, *, limit: int = ERROR_SNIPPET_CHARS) -> str:
    compact = "\n".join(line.rstrip() for line in text.strip().splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _wrapper_stream_detail(stdout: str, stderr: str) -> str:
    parts = []
    clean_stderr = _compact_text(_strip_remote_noise(stderr))
    clean_stdout = _compact_text(_strip_remote_noise(stdout))
    if clean_stderr:
        parts.append(f"stderr: {clean_stderr}")
    if clean_stdout:
        parts.append(f"stdout: {clean_stdout}")
    return "; ".join(parts)


def _exec_wrapper_failure_message(*, host: HostSpec, returncode: int, stdout: str, stderr: str) -> str:
    raw = "\n".join(part for part in (stderr, stdout) if part)
    detail = _wrapper_stream_detail(stdout, stderr)
    prefix = f"SSH failed before remote exec wrapper started on {host.name} (exit {returncode})"
    if returncode == 255:
        if "Stdio forwarding request failed" in raw or "UNKNOWN port 65535" in raw:
            return (
                f"{prefix}: ProxyJump/control-master forwarding was refused. "
                "The knuckles control master may be stale, or the target host may be unreachable. "
                "Check `ucl status {host}` or restart the knuckles master."
            ).format(host=host.name)
        if "No route to host" in raw:
            return f"{prefix}: target host is unreachable from the jump host. {detail}".strip()
        if "Connection refused" in raw:
            return f"{prefix}: SSH connection was refused. {detail}".strip()
        if "Permission denied" in raw:
            return f"{prefix}: SSH authentication failed. {detail}".strip()
        if "Could not resolve hostname" in raw or "Name or service not known" in raw:
            return f"{prefix}: hostname could not be resolved. {detail}".strip()
        if detail:
            return f"{prefix}: {detail}"
        return f"{prefix}: no stderr/stdout was returned; the host or jump connection is likely unreachable."
    if detail:
        return f"Remote exec wrapper failed before it could return a result on {host.name} (exit {returncode}): {detail}"
    return f"Remote exec wrapper failed before it could return a result on {host.name} (exit {returncode}); no stderr/stdout was returned."


def _parse_sync_exec_result(*, host: HostSpec, stdout: str, stderr: str) -> dict[str, Any]:
    try:
        raw_payload = _extract_between(stdout, EXEC_SENTINEL_BEGIN, EXEC_SENTINEL_END, label="exec")
    except RuntimeError as exc:
        detail = _wrapper_stream_detail(stdout, stderr)
        message = f"Remote exec wrapper on {host.name} did not return sentinel JSON"
        if detail:
            message += f": {detail}"
        else:
            message += "."
        raise RuntimeError(message) from exc
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        snippet = _compact_text(raw_payload)
        raise RuntimeError(f"Remote exec wrapper on {host.name} returned malformed sentinel JSON: {snippet}") from exc
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
    decoded_stdout = _decode_stream(result["stdout"])
    decoded_stderr = _decode_stream(result["stderr"])
    ok = int(result["returncode"]) == 0 and not result["timed_out"] and not result["wrapper_error"]
    return json.dumps(
        {
            "ok": ok,
            "host": host.name,
            "ssh_host": host.ssh_host,
            "command": list(command),
            "cwd": args.cwd,
            "timeout": None if args.timeout == 0 else args.timeout,
            "timed_out": result["timed_out"],
            "wrapper_error": result["wrapper_error"],
            "returncode": result["returncode"],
            "stdout": decoded_stdout,
            "stderr": decoded_stderr,
            "error": "" if ok else (decoded_stderr.strip() or f"remote command exited {result['returncode']}"),
        },
        indent=2,
        sort_keys=True,
    )


def _format_sync_exec_error_json(
    *,
    host: HostSpec,
    command: tuple[str, ...],
    args: argparse.Namespace,
    error: str,
    returncode: int = 2,
    stdout: str = "",
    stderr: str = "",
    wrapper_error: bool = True,
    timed_out: bool = False,
) -> str:
    return json.dumps(
        {
            "ok": False,
            "host": host.name,
            "ssh_host": host.ssh_host,
            "command": list(command),
            "cwd": args.cwd,
            "timeout": None if args.timeout == 0 else args.timeout,
            "timed_out": timed_out,
            "wrapper_error": wrapper_error,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
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
        _ssh_python_argv(host.ssh_host, connect_timeout=int(args.connect_timeout)),
        input=_sync_exec_source(params),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        message = _exec_wrapper_failure_message(
            host=host,
            returncode=int(getattr(proc, "returncode", 1)),
            stdout=getattr(proc, "stdout", "") or "",
            stderr=getattr(proc, "stderr", "") or "",
        )
        if args.json:
            print(
                _format_sync_exec_error_json(
                    host=host,
                    command=command,
                    args=args,
                    error=message,
                    returncode=int(getattr(proc, "returncode", 1)),
                    stdout=_strip_remote_noise(getattr(proc, "stdout", "") or ""),
                    stderr=_strip_remote_noise(getattr(proc, "stderr", "") or ""),
                )
            )
            return 2
        raise RuntimeError(message)
    try:
        result = _parse_sync_exec_result(
            host=host,
            stdout=getattr(proc, "stdout", "") or "",
            stderr=getattr(proc, "stderr", "") or "",
        )
    except RuntimeError as exc:
        if args.json:
            print(
                _format_sync_exec_error_json(
                    host=host,
                    command=command,
                    args=args,
                    error=str(exc),
                    stdout=_strip_remote_noise(getattr(proc, "stdout", "") or ""),
                    stderr=_strip_remote_noise(getattr(proc, "stderr", "") or ""),
                )
            )
            return 2
        raise
    if args.json:
        print(_format_sync_exec_json(host=host, command=command, args=args, result=result))
    else:
        if result["stdout"]:
            sys.stdout.write(_decode_stream(result["stdout"]))
        if result["wrapper_error"]:
            stderr_text = _decode_stream(result["stderr"])
            detail = stderr_text.strip() or "unknown wrapper error"
            sys.stderr.write(f"Remote exec wrapper on {host.name} failed before the command could run: {detail}\n")
        elif result["stderr"]:
            sys.stderr.write(_decode_stream(result["stderr"]))
    return int(result["returncode"])


def run_exec_multi_sync(
    args: argparse.Namespace,
    *,
    hosts: list[HostSpec],
    command: tuple[str, ...],
    stdin_body: str | None,
    runner=subprocess.run,
) -> int:
    if args.dry_run:
        print(
            "\n".join(
                [
                    "dry_run: true",
                    "command:    exec",
                    "mode:       multi-sync",
                    f"hosts:      {', '.join(host.name for host in hosts)}",
                    f"shell:      {args.shell}",
                    f"cwd:        {args.cwd or '-'}",
                    f"timeout:    {'none' if args.timeout == 0 else args.timeout:g}",
                    f"connect:    {'none' if args.connect_timeout == 0 else args.connect_timeout}",
                    f"stdin:      {'yes' if args.stdin else 'no'}",
                    f"argv:       {json.dumps(list(command))}",
                ]
            )
        )
        return 0
    ensure_knuckles_master(runner=runner)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(hosts), 1), 8)) as executor:
        future_to_host = {executor.submit(_multi_exec_one, host, args, command, stdin_body, runner=runner): host for host in hosts}
        by_host: dict[str, dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(future_to_host):
            row = future.result()
            by_host[row["host"]] = row
        rows = [by_host[host.name] for host in hosts]
    if args.json:
        print(json.dumps({"results": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            status = "ok" if row.get("ok") else "fail"
            first = (row.get("stdout") or row.get("stderr") or row.get("error") or "").strip().splitlines()
            print(f"{row['host']}: {status} rc={row.get('returncode')} {first[0] if first else ''}".rstrip())
    return 0 if all(row.get("ok") for row in rows) else 2


def run_run(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    if not args.session and not args.new_session:
        raise RuntimeError("ucl run requires --session NAME or --new-session")
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
            remote_root=args.remote_root,
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
        remote_root=args.remote_root,
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
        require_explicit_when_not_single=True,
    )
    upload_bundle(plan, runner=runner, popener=popener)
    launcher = write_launcher_files(plan, runner=runner)
    launch_tmux(plan, decision, launcher, runner=runner)
    provenance = _provenance_for_plan(plan, args=args, env=env, stdin_body=None)
    write_record(_record_from_plan(plan, decision, provenance=provenance))
    print(format_summary(plan, decision))
    return 0


def run_exec(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    hosts = _resolve_exec_hosts(args)
    host = hosts[0]
    command = _strip_remainder(args.exec_command)
    stdin_body = sys.stdin.read() if args.stdin else None
    if not args.detach:
        if len(hosts) > 1:
            return run_exec_multi_sync(args, hosts=hosts, command=command, stdin_body=stdin_body, runner=runner)
        return run_exec_sync(args, host=host, command=command, stdin_body=stdin_body, runner=runner)
    if len(hosts) != 1:
        raise RuntimeError("multi-host exec is synchronous only; remove --detach or run separate detached jobs")
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
            remote_root=args.remote_root,
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
        remote_root=args.remote_root,
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
    provenance = _provenance_for_plan(plan, args=args, env=env, stdin_body=stdin_body)
    write_record(_record_from_plan(plan, decision, provenance=provenance))
    print(format_summary(plan, decision))
    return 0


def _record_from_plan(plan, decision, *, provenance: dict[str, Any] | None = None) -> RunRecord:
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
        provenance=provenance or {},
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


def _ssh_python_argv(host: str, *, connect_timeout: int | None = None) -> list[str]:
    timeout = None if connect_timeout in (None, 0) else int(connect_timeout)
    return build_remote_python_argv(host, timeout_seconds=timeout)


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
    remote_root = args.remote_root or default_remote_root()
    ensure_knuckles_master(runner=runner)
    proc = runner(
        _ssh_python_argv(host.ssh_host),
        input=_clean_source(int(args.older_than_days), bool(args.execute), remote_root),
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


def _record_status(record: RunRecord, *, runner=subprocess.run, timeout_seconds: int = 8) -> dict[str, Any]:
    host = HostSpec(name=record.host, ssh_host=record.ssh_host)
    try:
        sessions = list_remote_sessions(host, runner=runner, timeout_seconds=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - status should report per-job failures.
        return {"status": "unreachable", "error": str(exc), "sessions": []}
    if record.session in sessions:
        return {"status": "running", "error": "", "sessions": list(sessions)}
    return {"status": "exited_or_missing", "error": "", "sessions": list(sessions)}


def _record_payload(record: RunRecord, *, runner=subprocess.run, timeout_seconds: int = 8) -> dict[str, Any]:
    status = _record_status(record, runner=runner, timeout_seconds=timeout_seconds)
    provenance = record.provenance or {}
    return {
        "run_id": record.run_id,
        "kind": record.kind,
        "host": record.host,
        "ssh_host": record.ssh_host,
        "session": record.session,
        "window": record.window,
        "remote_dir": record.remote_dir,
        "log_path": record.log_path,
        "command": list(record.command),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "project": provenance.get("project", ""),
        "selected_gpu": provenance.get("selected_gpu", ""),
        "provenance": provenance,
        **status,
    }


def _format_jobs_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "RUN_ID  STATUS  PROJECT  HOST  GPU  SESSION  WINDOW  KIND  UPDATED"
    columns = ("RUN_ID", "STATUS", "PROJECT", "HOST", "GPU", "SESSION", "WINDOW", "KIND", "UPDATED")
    data = [
        [
            row.get("run_id", ""),
            row.get("status", ""),
            row.get("project", ""),
            row.get("host", ""),
            row.get("selected_gpu", ""),
            row.get("session", ""),
            row.get("window", ""),
            row.get("kind", ""),
            row.get("updated_at") or row.get("created_at") or "",
        ]
        for row in rows
    ]
    widths = [len(col) for col in columns]
    for row in data:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]
    lines = ["  ".join(col.ljust(width) for col, width in zip(columns, widths))]
    lines.extend("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in data)
    return "\n".join(lines)


def run_jobs(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    ensure_knuckles_master(runner=runner)
    rows = [_record_payload(record, runner=runner, timeout_seconds=args.timeout_seconds) for record in list_records()]
    if args.json:
        print(json.dumps({"jobs": rows}, indent=2, sort_keys=True))
    else:
        print(_format_jobs_table(rows))
    return 0


def run_info(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    ensure_knuckles_master(runner=runner)
    row = _record_payload(read_record(args.run_ref), runner=runner, timeout_seconds=args.timeout_seconds)
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        for key in (
            "run_id",
            "status",
            "project",
            "kind",
            "host",
            "selected_gpu",
            "session",
            "window",
            "remote_dir",
            "log_path",
            "command",
            "created_at",
            "updated_at",
            "provenance",
            "error",
        ):
            value = row.get(key)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            print(f"{key}: {value}")
    return 0


def _stop_source(session: str, window: str, signal: str) -> str:
    return f"""
import json
import os
import signal as signal_mod
import subprocess
SESSION={json.dumps(session)}
WINDOW={json.dumps(window)}
SIGNAL={json.dumps(signal)}
target = SESSION + ":" + WINDOW
sig = signal_mod.SIGKILL if SIGNAL == "KILL" else signal_mod.SIGTERM
pane = subprocess.run(["tmux", "display-message", "-p", "-t", target, "#{{pane_pid}}"], capture_output=True, text=True)
signal_error = ""
if pane.returncode == 0 and pane.stdout.strip().isdigit():
    pid = int(pane.stdout.strip())
    try:
        os.killpg(os.getpgid(pid), sig)
    except Exception as exc:
        signal_error = f"{{type(exc).__name__}}: {{exc}}"
proc = subprocess.run(["tmux", "kill-window", "-t", target], capture_output=True, text=True)
if proc.returncode != 0 and "can't find window" in proc.stderr:
    proc = subprocess.run(["tmux", "kill-session", "-t", SESSION], capture_output=True, text=True)
print(json.dumps({{"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "signal_error": signal_error}}, sort_keys=True))
raise SystemExit(proc.returncode)
"""


def run_stop(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    if args.run_ref == "last" and not args.yes:
        raise RuntimeError("refusing to stop 'last' without --yes; pass an explicit run id or use 'ucl stop last --yes'")
    record = read_record(args.run_ref)
    ensure_knuckles_master(runner=runner)
    proc = runner(
        _ssh_python_argv(record.ssh_host),
        input=_stop_source(record.session, record.window, args.signal),
        capture_output=True,
        text=True,
        shell=False,
    )
    payload: dict[str, Any] = {
        "run_id": record.run_id,
        "host": record.host,
        "session": record.session,
        "window": record.window,
        "returncode": int(getattr(proc, "returncode", 1)),
        "stdout": getattr(proc, "stdout", "") or "",
        "stderr": getattr(proc, "stderr", "") or "",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_id: {record.run_id}")
        print(f"target: {record.session}:{record.window}")
        print("status: stopped" if payload["returncode"] == 0 else "status: stop_failed")
        if payload["stderr"]:
            print(payload["stderr"], file=sys.stderr, end="")
    return 0 if payload["returncode"] == 0 else 2


def _copy_endpoint_manifest(endpoint: copy_tools.Endpoint, *, verify: str, runner=subprocess.run) -> dict[str, Any]:
    return copy_tools.read_manifest(endpoint, sha256=(verify == "sha256"), runner=runner)


def run_copy(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    src = copy_tools.resolve_endpoint(copy_tools.parse_endpoint(args.src), args.catalog)
    dst = copy_tools.resolve_endpoint(copy_tools.parse_endpoint(args.dst), args.catalog)
    needs_ssh = bool(src.host or dst.host)
    if needs_ssh and not args.dry_run:
        ensure_knuckles_master(runner=runner)
    before = None
    if args.verify != "none" and not args.dry_run:
        before = _copy_endpoint_manifest(src, verify=args.verify, runner=runner)
    if src.host and dst.host:
        mode = "remote-to-remote"
        argv = copy_tools.build_remote_to_remote_argv(
            src,
            dst,
            partial=args.partial,
            progress=args.progress,
            dry_run=args.dry_run,
            rsync_args=tuple(args.rsync_args),
        )
    else:
        mode = "rsync"
        argv = copy_tools.build_rsync_argv(
            src,
            dst,
            partial=args.partial,
            progress=args.progress,
            dry_run=args.dry_run,
            rsync_args=tuple(args.rsync_args),
        )
    if args.dry_run:
        payload = {
            "src": args.src,
            "dst": args.dst,
            "resolved_src": src.rsync_spec(),
            "resolved_dst": dst.rsync_spec(),
            "mode": mode,
            "argv": argv,
            "dry_run": True,
            "verify": args.verify,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(" ".join(shlex.quote(part) for part in argv))
        return 0
    proc = runner(argv, capture_output=True, text=True, shell=False)
    ok = int(getattr(proc, "returncode", 1)) == 0
    verify_payload: dict[str, Any] = {"mode": args.verify, "ok": None, "message": "skipped"}
    if ok and args.verify != "none" and before is not None:
        after = _copy_endpoint_manifest(dst, verify=args.verify, runner=runner)
        verify_ok, message = copy_tools.compare_manifests(before, after, sha256=(args.verify == "sha256"))
        verify_payload = {"mode": args.verify, "ok": verify_ok, "message": message}
        ok = ok and verify_ok
    payload = {
        "ok": ok,
        "src": args.src,
        "dst": args.dst,
        "resolved_src": src.rsync_spec(),
        "resolved_dst": dst.rsync_spec(),
        "mode": mode,
        "argv": argv,
        "returncode": int(getattr(proc, "returncode", 1)),
        "stdout": _strip_remote_noise(getattr(proc, "stdout", "") or ""),
        "stderr": _strip_remote_noise(getattr(proc, "stderr", "") or ""),
        "verify": verify_payload,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if payload["stdout"]:
            print(payload["stdout"], end="")
        if payload["stderr"]:
            print(payload["stderr"], file=sys.stderr, end="")
        print(f"mode: {mode}")
        print(f"argv: {shlex.join(argv)}")
        print(f"returncode: {payload['returncode']}")
        print(f"verify: {verify_payload['message']}")
    return 0 if ok else 2


def run_env(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    ensure_knuckles_master(runner=runner)
    gpu = args.gpu
    if gpu == "auto":
        if float(args.min_free_vram_gb) < 0:
            raise ValueError("--min-free-vram-gb must be >= 0")
        rows = inventory.collect([host], runner=runner, jobs=1, min_free_vram_gb=float(args.min_free_vram_gb))
        gpu = _best_free_gpu(rows[0], min_free_vram_gb=float(args.min_free_vram_gb))
    payload = envcheck.run_env_check(host, remote_root=args.remote_root, create=args.create, gpu=gpu, runner=runner)
    payload.update({"host": host.name, "ssh_host": host.ssh_host})
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"host: {host.name}")
        print(f"remote_root: {payload['remote_root']}")
        print(f"root_exists: {payload['root_exists']}")
        print(f"tmp_free_gb: {payload['tmp_free_gb']}")
        print(f"cuda_visibility: {'yes' if payload['cuda_visibility_exists'] else 'no'}")
        print(f"python_setup: {'yes' if payload['python_setup_exists'] else 'no'}")
        print(f"ok: {payload['ok']}")
    return 0 if payload.get("ok") else 2


def _multi_exec_one(host: HostSpec, args: argparse.Namespace, command: tuple[str, ...], stdin_body: str | None, *, runner=subprocess.run) -> dict[str, Any]:
    timeout_value = float(getattr(args, "timeout_seconds", getattr(args, "timeout", 60.0)))
    connect_timeout = int(getattr(args, "connect_timeout", 30))
    local_args = argparse.Namespace(
        dry_run=False,
        stdin=args.stdin,
        shell=args.shell,
        timeout=timeout_value,
        cwd=getattr(args, "cwd", None),
        json=False,
        gpu=args.gpu,
        min_free_vram_gb=getattr(args, "min_free_vram_gb", DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB),
        env=getattr(args, "env", []),
    )
    try:
        env = _resolve_env(local_args, host, runner=runner)
        params = {
            "mode": "stdin" if args.stdin else "command",
            "argv": list(command),
            "stdin_b64": base64.b64encode((stdin_body or "").encode("utf-8")).decode("ascii"),
            "env": dict(env),
            "shell": args.shell,
            "cwd": getattr(args, "cwd", None),
            "timeout": timeout_value,
        }
        proc = runner(
            _ssh_python_argv(host.ssh_host, connect_timeout=connect_timeout),
            input=_sync_exec_source(params),
            capture_output=True,
            text=True,
            shell=False,
        )
        if int(getattr(proc, "returncode", 1)) != 0:
            return {
                "host": host.name,
                "ok": False,
                "returncode": int(getattr(proc, "returncode", 1)),
                "stdout": _strip_remote_noise(getattr(proc, "stdout", "") or ""),
                "stderr": _strip_remote_noise(getattr(proc, "stderr", "") or ""),
                "error": _exec_wrapper_failure_message(host=host, returncode=int(getattr(proc, "returncode", 1)), stdout=getattr(proc, "stdout", "") or "", stderr=getattr(proc, "stderr", "") or ""),
            }
        result = _parse_sync_exec_result(host=host, stdout=getattr(proc, "stdout", "") or "", stderr=getattr(proc, "stderr", "") or "")
        return {
            "host": host.name,
            "ok": int(result["returncode"]) == 0 and not result["timed_out"] and not result["wrapper_error"],
            "returncode": int(result["returncode"]),
            "stdout": _decode_stream(result["stdout"]),
            "stderr": _decode_stream(result["stderr"]),
            "timed_out": result["timed_out"],
            "wrapper_error": result["wrapper_error"],
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - multi-host exec reports per-host errors.
        return {"host": host.name, "ok": False, "returncode": 2, "stdout": "", "stderr": "", "error": str(exc)}


def _clean_source(days: int, execute: bool, remote_root: str) -> str:
    return f"""
import json
import os
import shutil
import time
BEGIN={json.dumps(CLEAN_SENTINEL_BEGIN)}
END={json.dumps(CLEAN_SENTINEL_END)}
ROOT={json.dumps(remote_root)}
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
    if raw_argv and raw_argv[0] == "copy":
        try:
            args = _parse_copy_argv(raw_argv[1:])
        except SystemExit as exc:
            return int(exc.code or 0)
        try:
            return run_copy(args, runner=runner)
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
        if args.command == "jobs":
            return run_jobs(args, runner=runner)
        if args.command == "info":
            return run_info(args, runner=runner)
        if args.command == "stop":
            return run_stop(args, runner=runner)
        if args.command == "copy":
            return run_copy(args, runner=runner)
        if args.command == "env":
            return run_env(args, runner=runner)
    except Exception as exc:  # noqa: BLE001 - CLI should render concise failures.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
