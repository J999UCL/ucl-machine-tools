"""Unified CLI for UCL machine tools."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import replace
import hashlib
import json
import math
import posixpath
import subprocess
import sys
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from ucl_machine_tools.hosts import HostSpec, load_catalog, parse_selector
from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import envcheck
from ucl_machine_tools import inventory
from ucl_machine_tools import job_control
from ucl_machine_tools.launch import (
    RemoteJobPlan,
    build_exec_plan,
    build_run_plan,
    build_staged_run_plan,
    create_remote_dir,
    decide_tmux,
    default_remote_root,
    format_summary,
    launch_tmux,
    list_remote_sessions,
    normalize_remote_command,
    parse_env,
    upload_bundle,
    utc_run_id,
    write_launcher_files,
)
from ucl_machine_tools.registry import RunRecord, list_records, read_record, utc_now, write_record
from ucl_machine_tools.ssh import build_remote_python_argv, ensure_knuckles_master
from ucl_machine_tools import stage as stage_tools
from ucl_machine_tools import stage_registry
from ucl_machine_tools.stage_registry import StageRecord
from ucl_machine_tools.uv_project import (
    RemoteUvLayout,
    UvProjectSpec,
    derive_remote_layout,
    hash_setup_environment,
    load_uv_project,
    materialize_source_snapshot,
)
from ucl_machine_tools.uv_remote import (
    UvRemotePaths,
    UvSetupSpec,
    build_setup_payload,
    parse_state_json,
)

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
TAIL_STOP_TIMEOUT_SECONDS = 5.0
DEFAULT_STATUS_JOBS = 32
DEFAULT_STATUS_TIMEOUT_SECONDS = 5


class JobIdentityUnreachable(RuntimeError):
    """The host could not be reached while probing a recorded job."""


class JobIdentityProbeError(RuntimeError):
    """The host responded, but its job identity could not be verified."""


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
    ucl stage --uv --host barbury-l --name demo --local-dir ./project --remote-root /tmp/thakwani/demo --gpu auto
    ucl run --stage STAGE_ID --script scripts/train.sh --new-session --gpu auto
    ucl exec barbury-l --detach --new-session -- hostname
    ucl run --host barbury-l --new-session --gpu auto --min-free-vram-gb 20 --local-dir ./bundle --script run.sh
    ucl jobs
    ucl info last
    ucl tail last --live
    ucl fetch last
    ucl stop RUN_ID
    ucl clean barbury-l

  Copy data:
    ucl copy /tmp/a barbury-l:/tmp/a --verify size
    ucl copy barbury-l:/tmp/a barnacle-l:/tmp/a -- --partial --info=progress2 --exclude '*.pt'
    ucl copy cream:/tmp/downloads brent-l:/tmp/downloads --verify sha256 --partial --retries 2
    ucl copy cream:/tmp/downloads brent-l:/tmp/clean --verify sha256 --reuse-from brent-l:/tmp/existing

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

    stage = subparsers.add_parser(
        "stage",
        help="Upload a locked UV project and prepare its remote environment.",
        description=(
            "Validate pyproject.toml, uv.lock, and .python-version; upload an ignore-aware "
            "content-addressed source snapshot; then prepare the exact locked environment asynchronously in tmux."
        ),
    )
    stage.add_argument("--uv", action="store_true", required=True, help="require the locked UV staging workflow")
    stage.add_argument("--host", required=True)
    stage.add_argument("--name", help="safe stage name; defaults to the local directory name")
    stage.add_argument("--local-dir", required=True, type=Path)
    stage.add_argument("--remote-root", required=True, help="absolute managed project root on the remote host")
    stage.add_argument("--catalog", type=Path)
    stage.add_argument("--env", action="append", default=[], help="setup env KEY=VALUE; values are not stored in records")
    stage.add_argument("--gpu", help="GPU id or auto")
    stage.add_argument("--min-free-vram-gb", type=float, default=DEFAULT_AUTO_GPU_MIN_FREE_VRAM_GB)
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--json", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="Launch a local bundle or a verified stage in tmux.",
        description=(
            "Upload and launch a local bundle, or use --stage ID to launch from a ready immutable UV stage "
            "without uploading source or syncing dependencies."
        ),
    )
    run.add_argument("--host")
    run.add_argument("--local-dir", type=Path)
    run.add_argument("--stage", help="verified stage id from `ucl stage`")
    run.add_argument("--script", required=True)
    _add_launch_common_flags(run)
    run.add_argument("--arg", action="append", default=[], help="script argument; repeat for multiple args")
    run.add_argument("--replace", action="store_true", help="replace an existing non-empty remote bundle dir")

    exec_parser = subparsers.add_parser("exec", help="Run a small remote command or snippet.")
    _configure_exec_parser(exec_parser)

    tail = subparsers.add_parser("tail", help="Print or live-stream a recorded run log.")
    tail.add_argument("run_ref", nargs="?", default="last")
    tail.add_argument("--lines", type=int, default=80, help="initial log lines to show (default: 80)")
    tail.add_argument(
        "--live",
        "--follow",
        dest="live",
        action="store_true",
        help="stream new log lines until interrupted with Ctrl-C",
    )

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
    stop.add_argument(
        "--grace-seconds",
        type=float,
        default=5.0,
        help="seconds to wait for recorded processes to exit after signaling (default: 5)",
    )
    stop.add_argument("--yes", action="store_true", help="allow stopping the latest recorded run via 'last'")
    stop.add_argument("--json", action="store_true")
    stop.add_argument("--timeout-seconds", type=int, default=8)

    copy = subparsers.add_parser("copy", help="Copy or reconcile local/remote paths with rsync.")
    copy.add_argument("src")
    copy.add_argument("dst")
    copy.add_argument("--catalog", type=Path)
    copy.add_argument("--verify", choices=("size", "sha256", "none"), default="none")
    copy.add_argument("--reuse-from", help="same-destination-host tree containing files eligible for hard-link reuse")
    copy.add_argument("--retries", type=int, default=1, help="verification retries after the first transfer (default: 1)")
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


def _validate_run_mode(args: argparse.Namespace) -> str:
    if args.stage:
        conflicts = (
            ("--host", args.host),
            ("--local-dir", args.local_dir),
            ("--remote-dir", args.remote_dir),
            ("--remote-root", args.remote_root),
            ("--replace", args.replace),
        )
        for flag, value in conflicts:
            if value:
                raise ValueError(f"--stage cannot be combined with {flag}")
        return "stage"
    if not args.host:
        raise ValueError("ordinary ucl run requires --host")
    if args.local_dir is None:
        raise ValueError("ordinary ucl run requires --local-dir")
    return "bundle"


def _add_inventory_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selector", help="explicit selector; overrides positional target")
    parser.add_argument("--catalog", type=Path, help="host catalog JSON")
    parser.add_argument("--root", default="/tmp/ucl-machine-tools", help="remote scratch root to check")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--table", action="store_true", help="emit a human table")
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_STATUS_JOBS,
        help=f"maximum concurrent probes (default: {DEFAULT_STATUS_JOBS})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_STATUS_TIMEOUT_SECONDS,
        help=f"per-host SSH handshake timeout (default: {DEFAULT_STATUS_TIMEOUT_SECONDS})",
    )
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
        description="Copy paths with rsync, or reconcile them with pre/post verification.",
        epilog=(
            "Use '--verify sha256' to hash both sides, skip exact files, transfer only missing/mismatched files, "
            "and retry verification failures. Remote copies use a framed SSH transport that removes only startup "
            "output before rsync begins. Raw rsync args after '--' are available only without --verify, but cannot "
            "replace the protected transport or inject remote-side rsync options."
        ),
    )
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--verify",
        choices=("size", "sha256", "none"),
        default="none",
        help="pre-compare, selectively transfer, and verify by size or SHA-256",
    )
    parser.add_argument(
        "--reuse-from",
        help="same-destination-host directory whose exact files should be hard-linked into DST",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="verification retries after the first selective transfer (default: 1)",
    )
    parser.add_argument("--partial", action="store_true", help="preserve partial data for resumable transfers")
    parser.add_argument("--progress", action="store_true", help="add rsync --info=progress2")
    parser.add_argument("--dry-run", action="store_true", help="inspect the transfer plan without modifying files")
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
        "reuse_from": None,
        "retries": 1,
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
        "--reuse-from": "reuse_from",
        "--retries": "retries",
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
            elif key == "retries":
                try:
                    values[key] = int(raw_value)
                except ValueError:
                    parser.error("--retries must be a non-negative integer")
                if values[key] < 0:
                    parser.error("--retries must be a non-negative integer")
            else:
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
    stream_human = not (args.json and not args.table) and mode != "recommend"
    if stream_human:
        print(inventory.format_stream_header(), flush=True)

    def emit_row(row: dict[str, Any]) -> None:
        filtered = _filter_status_rows(mode, args, [row])
        if filtered:
            print(inventory.format_stream_row(filtered[0]), flush=True)

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
        on_result=emit_row if stream_human else None,
    )
    rows = _filter_status_rows(mode, args, rows)
    if args.json and not args.table:
        print(json.dumps(inventory.to_jsonable(rows), indent=2, sort_keys=True))
    elif not stream_human:
        print(inventory.format_table(rows))
    return 0


def run_doctor(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    ensure_knuckles_master(runner=runner)
    row = inventory.collect([host], runner=runner, jobs=1, timeout_seconds=args.timeout_seconds)[0]
    print(f"host:          {host.name}")
    print(f"status:        {row.get('status')}")
    print(f"tmp_scratch:  {'yes' if (row.get('scratch') or {}).get('exists') else 'no'}")
    if not row.get("ok"):
        errors = row.get("errors") or ["remote probe failed"]
        print("tmux_sessions: unavailable")
        print(f"error:         {errors[0]}")
        return 2
    sessions = list_remote_sessions(host, runner=runner, timeout_seconds=args.timeout_seconds)
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
    clean_stderr = _compact_text(stderr)
    clean_stdout = _compact_text(stdout)
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
                    stdout=getattr(proc, "stdout", "") or "",
                    stderr=getattr(proc, "stderr", "") or "",
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
                    stdout=getattr(proc, "stdout", "") or "",
                    stderr=getattr(proc, "stderr", "") or "",
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


def _failed_state_path(ready_state_path: str) -> str:
    path = PurePosixPath(ready_state_path)
    return str(path.with_name(f"{path.stem}.failed{path.suffix}"))


def _validate_stage_state(record: StageRecord, payload: dict[str, object]) -> None:
    status = payload.get("status")
    if status == "missing":
        raise RuntimeError(f"stage state is missing: {record.stage_id}")
    if status == "invalid":
        raise RuntimeError(f"stage state is invalid: {payload.get('error') or record.stage_id}")
    raw_state = payload.get("state")
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"stage state is {status or 'unavailable'}: {record.stage_id}")
    state = parse_state_json(json.dumps(raw_state))
    if not state.ok:
        raise RuntimeError(f"stage failed during {state.phase}: {state.error}")
    expected = {
        "uv_version": record.uv_version,
        "source_sha256": record.source_hash,
        "lock_sha256": record.lock_hash,
        "setup_environment_sha256": record.setup_environment_hash,
        "python_request": record.python_request,
        "source_dir": record.source_path,
        "environment_dir": record.environment_path,
        "uv_binary_path": record.uv_path,
        "ready_state_path": record.state_path,
        "failed_state_path": _failed_state_path(record.state_path),
    }
    for field, value in expected.items():
        if getattr(state, field) != value:
            raise RuntimeError(f"stage state identity mismatch for {field}: {record.stage_id}")
    missing = payload.get("missing_paths")
    if isinstance(missing, list) and missing:
        raise RuntimeError(f"stage files are missing on {record.host}: {', '.join(str(item) for item in missing)}")


def _stage_summary(record: StageRecord, *, source_action: str, json_output: bool) -> None:
    payload = {
        "stage_id": record.stage_id,
        "host": record.host,
        "status": record.status,
        "source_action": source_action,
        "source_path": record.source_path,
        "environment_path": record.environment_path,
        "state_path": record.state_path,
        "setup_run_id": record.setup_run_id,
    }
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"stage_id:     {record.stage_id}")
    print(f"host:         {record.host}")
    print(f"status:       {record.status}")
    print(f"source:       {source_action}")
    print(f"source_dir:   {record.source_path}")
    print(f"environment:  {record.environment_path}")
    print(f"state:        {record.state_path}")
    if record.setup_run_id:
        print(f"setup_run:    {record.setup_run_id}")
        print(f"tail:         ucl tail {record.setup_run_id} --live")
    print(f"run_when_ready: ucl run --stage {record.stage_id} --script SCRIPT --new-session")


def _reconcile_and_launch_stage(
    *,
    host: HostSpec,
    layout: RemoteUvLayout,
    project: UvProjectSpec,
    base_record: StageRecord,
    source_reused: bool,
    setup_env: tuple[tuple[str, str], ...],
    gpu_id: str | None,
    cuda_visibility_script: str | None,
    failed_state: str,
    python_install_dir: str,
    json_output: bool,
    runner: Any,
) -> int:
    """Reconcile one stage identity and launch at most one local setup request."""

    try:
        existing_record = stage_registry.read_record(layout.stage_id)
    except ValueError as error:
        if "not found" not in str(error):
            raise
        existing_record = None
    state_payload = dict(
        stage_tools.probe_stage_state(
            host,
            ready_state_path=str(layout.state_file),
            failed_state_path=failed_state,
            runner=runner,
        )
    )
    if state_payload.get("status") == "ready":
        identity_record = existing_record or base_record
        _validate_stage_state(identity_record, state_payload)
        if existing_record is None:
            stage_registry.write_record(base_record)
        ready_record = stage_registry.update_status(
            identity_record.stage_id,
            "ready",
            provenance={"source_reused": source_reused, "environment_reused": True},
        )
        _stage_summary(
            ready_record,
            source_action="reused" if source_reused else "uploaded",
            json_output=json_output,
        )
        return 0
    if state_payload.get("status") == "invalid":
        raise RuntimeError(
            f"existing stage state is invalid: {state_payload.get('error') or layout.stage_id}"
        )

    if existing_record is not None and existing_record.status == "preparing" and existing_record.setup_run_id:
        sessions = list_remote_sessions(host, runner=runner)
        if existing_record.setup_run_id in sessions:
            _stage_summary(
                existing_record,
                source_action="reused" if source_reused else "uploaded",
                json_output=json_output,
            )
            return 0

    setup_run_id = utc_run_id(f"uvsetup_{layout.stage_name}")
    setup_remote_dir = str(layout.launchers_dir / setup_run_id)
    setup_log = posixpath.join(setup_remote_dir, "setup.log")
    paths = UvRemotePaths(
        source_dir=str(layout.source_dir),
        environment_dir=str(layout.environment_dir),
        uv_cache_dir=str(layout.uv_cache_dir),
        uv_tool_dir=str(layout.uv_tools_dir),
        uv_binary_path=str(layout.uv_binary),
        python_install_dir=python_install_dir,
        ready_state_path=str(layout.state_file),
        failed_state_path=failed_state,
        log_path=setup_log,
        environment_lock_path=str(layout.state_dir / "locks" / f"env-{layout.environment_id}.lock"),
        uv_tool_lock_path=str(layout.remote_root / "state" / "locks" / f"uv-{layout.uv_version}.lock"),
    )
    spec = UvSetupSpec(
        uv_version=layout.uv_version,
        paths=paths,
        source_sha256=layout.source_sha256,
        lock_sha256=layout.lock_sha256,
        setup_environment_sha256=layout.setup_environment_sha256,
        python_request=project.contract.python_request,
        gpu_id=gpu_id,
        cuda_visibility_script=cuda_visibility_script,
        setup_env=setup_env,
    )
    payload = build_setup_payload(
        spec,
        csh_driver_path=posixpath.join(setup_remote_dir, ".ucl_uv_setup.csh"),
        bash_driver_path=posixpath.join(setup_remote_dir, ".ucl_uv_setup.sh"),
    )
    setup_plan = RemoteJobPlan(
        kind="stage-setup",
        host=host,
        run_id=setup_run_id,
        remote_dir=setup_remote_dir,
        remote_root=str(layout.launchers_dir),
        log_path=setup_log,
        work_dir=str(layout.source_dir),
        command=payload.entrypoint,
        env=(),
        shell="bash",
        requested_session=setup_run_id,
        new_session=True,
        window="uv_setup",
    )
    sessions = list_remote_sessions(host, runner=runner)
    decision = decide_tmux(
        sessions=sessions,
        generated_session=setup_run_id,
        requested_session=setup_run_id,
        new_session=True,
        window=setup_plan.window,
    )
    create_remote_dir(setup_plan, runner=runner)
    stage_tools.write_setup_payload(host, payload, runner=runner)
    record_seed = existing_record or base_record
    record = stage_registry.write_record(
        replace(
            record_seed,
            setup_run_id=setup_run_id,
            status="preparing",
            provenance={
                **record_seed.provenance,
                "source_reused": source_reused,
                "selected_gpu": gpu_id or "",
            },
        )
    )
    provisional = _record_from_plan(
        setup_plan,
        decision,
        provenance={
            "project": layout.stage_name,
            "stage_id": layout.stage_id,
            "source_hash": layout.source_sha256,
            "lock_hash": layout.lock_sha256,
            "selected_gpu": gpu_id or "",
        },
        identity={"pending_launch": True},
    )
    write_record(provisional)
    try:
        identity = launch_tmux(
            setup_plan,
            decision,
            PurePosixPath(payload.csh_driver_path).name,
            runner=runner,
        )
    except Exception:
        stage_registry.update_status(record.stage_id, "launch_failed")
        raise
    write_record(replace(provisional, identity=identity))
    _stage_summary(
        record,
        source_action="reused" if source_reused else "uploaded",
        json_output=json_output,
    )
    return 0


def run_stage(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    host = _resolve_one_host(args.host, catalog_path=args.catalog)
    project = load_uv_project(args.local_dir, runner=runner)
    requested_setup_env = parse_env(args.env)
    setup_identity_context = (("gpu_request", args.gpu or "none"),)
    setup_environment_hash = hash_setup_environment(
        requested_setup_env,
        context=setup_identity_context,
    )
    stage_name = args.name or project.contract.root.name
    layout = derive_remote_layout(
        remote_root=args.remote_root,
        stage_name=stage_name,
        host=host.name,
        uv_version=project.uv.version,
        lock_sha256=project.lock_sha256,
        source_sha256=project.source_sha256,
        setup_environment_sha256=setup_environment_hash,
    )
    failed_state = str(layout.state_dir / f"{layout.stage_id}.failed.json")
    python_install_dir = str(layout.python_install_dir)
    base_record = StageRecord(
        stage_id=layout.stage_id,
        name=layout.stage_name,
        host=host.name,
        ssh_host=host.ssh_host,
        remote_root=str(layout.remote_root),
        source_path=str(layout.source_dir),
        environment_path=str(layout.environment_dir),
        uv_path=str(layout.uv_binary),
        cache_path=str(layout.uv_cache_dir),
        source_hash=project.source_sha256,
        lock_hash=project.lock_sha256,
        setup_environment_hash=setup_environment_hash,
        uv_version=project.uv.version,
        python_request=project.contract.python_request,
        python_path=str(layout.environment_dir / "bin" / "python"),
        state_path=str(layout.state_file),
        status="planned" if args.dry_run else "preparing",
        provenance={
            "local_dir": str(project.contract.root),
            "local_git_sha": _git_sha(project.contract.root),
            "uv_executable": str(project.uv.executable),
            "manifest_files": len(project.manifest.entries),
            "manifest_bytes": project.manifest.total_bytes,
            "setup_environment_hash": setup_environment_hash,
            "setup_env_keys": sorted(key for key, _ in requested_setup_env),
        },
    )
    if args.dry_run:
        _stage_summary(base_record, source_action="not_checked", json_output=args.json)
        return 0

    ensure_knuckles_master(runner=runner)
    resolved_env = _resolve_env(args, host, runner=runner)
    gpu_id = _selected_gpu(resolved_env) or None
    setup_env = tuple((key, value) for key, value in resolved_env if key != "CUDA_VISIBLE_DEVICES")
    if dict(setup_env) != dict(requested_setup_env):
        raise RuntimeError("resolved setup environment changed after stage identity was derived")
    remote_environment = envcheck.run_env_check(
        host,
        remote_root=str(layout.remote_root),
        create=False,
        gpu=gpu_id,
        runner=runner,
    )
    if not remote_environment.get("python_setup_exists"):
        raise RuntimeError(f"TSG Python setup is unavailable on {host.name}")
    if gpu_id is not None and not remote_environment.get("cuda_visibility_exists"):
        raise RuntimeError(f"CUDA visibility setup is unavailable on {host.name}")
    cuda_visibility_script = (
        str(remote_environment["cuda_visibility_script"]) if gpu_id is not None else None
    )
    stage_tools.verify_managed_paths(
        host,
        remote_root=str(layout.remote_root),
        managed_paths=tuple(
            str(path)
            for path in (
                layout.sources_dir,
                layout.environments_dir,
                layout.uv_tools_dir,
                layout.python_install_dir,
                layout.uv_cache_dir,
                layout.state_dir,
                layout.launchers_dir,
            )
        ),
        runner=runner,
    )

    snapshot = materialize_source_snapshot(project.manifest)
    try:
        source_result = stage_tools.sync_source_snapshot(
            host,
            manifest=snapshot.manifest,
            source_dir=str(layout.source_dir),
            sources_dir=str(layout.sources_dir),
            runner=runner,
        )
    finally:
        snapshot.cleanup()

    with stage_registry.claim_stage(layout.stage_id):
        return _reconcile_and_launch_stage(
            host=host,
            layout=layout,
            project=project,
            base_record=base_record,
            source_reused=source_result.reused,
            setup_env=setup_env,
            gpu_id=gpu_id,
            cuda_visibility_script=cuda_visibility_script,
            failed_state=failed_state,
            python_install_dir=python_install_dir,
            json_output=args.json,
            runner=runner,
        )


def run_run(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    mode = _validate_run_mode(args)
    if not args.session and not args.new_session:
        raise RuntimeError("ucl run requires --session NAME or --new-session")
    if mode == "stage":
        record = stage_registry.read_record(args.stage)
        host = _resolve_one_host(record.host, catalog_path=args.catalog)
        if host.ssh_host != record.ssh_host:
            raise RuntimeError(f"stage host catalog identity changed: {record.host}")
        if args.dry_run:
            env = parse_env(args.env)
        else:
            ensure_knuckles_master(runner=runner)
            stage_tools.verify_registered_source(
                host,
                source_dir=record.source_path,
                source_sha256=record.source_hash,
                runner=runner,
            )
            state_payload = dict(
                stage_tools.probe_stage_state(
                    host,
                    ready_state_path=record.state_path,
                    failed_state_path=_failed_state_path(record.state_path),
                    required_script=args.script,
                    runner=runner,
                )
            )
            try:
                _validate_stage_state(record, state_payload)
            except RuntimeError:
                if state_payload.get("status") == "failed":
                    stage_registry.update_status(record.stage_id, "failed")
                raise
            stage_tools.verify_stage_environment(
                host,
                source_dir=record.source_path,
                environment_dir=record.environment_path,
                uv_binary_path=record.uv_path,
                uv_cache_dir=record.cache_path,
                python_install_dir=posixpath.join(record.remote_root, "tools", "python"),
                python_request=record.python_request,
                runner=runner,
            )
            stage_registry.update_status(record.stage_id, "ready")
            env = _resolve_env(args, host, runner=runner)
        launcher_root = posixpath.join(record.remote_root, "launchers")
        plan = build_staged_run_plan(
            host=host,
            source_dir=record.source_path,
            environment_dir=record.environment_path,
            uv_bin=record.uv_path,
            uv_cache_dir=record.cache_path,
            python_install_dir=posixpath.join(record.remote_root, "tools", "python"),
            script=args.script,
            args=tuple(args.arg),
            env=env,
            shell=args.shell,
            session=args.session,
            new_session=args.new_session,
            window=args.window,
            remote_root=launcher_root,
            log_path=args.log,
        )
        if args.dry_run:
            print(_dry_run_summary(plan, subcommand="run --stage"))
            return 0
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
        provenance = {
            **_provenance_for_plan(plan, args=args, env=env, stdin_body=None),
            "stage_id": record.stage_id,
            "source_hash": record.source_hash,
            "lock_hash": record.lock_hash,
        }
        provisional = _record_from_plan(
            plan,
            decision,
            provenance=provenance,
            identity={"pending_launch": True},
        )
        write_record(provisional)
        identity = launch_tmux(plan, decision, launcher, runner=runner)
        write_record(replace(provisional, identity=identity))
        print(format_summary(plan, decision))
        return 0

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
    provenance = _provenance_for_plan(plan, args=args, env=env, stdin_body=None)
    provisional = _record_from_plan(
        plan,
        decision,
        provenance=provenance,
        identity={"pending_launch": True},
    )
    write_record(provisional)
    identity = launch_tmux(plan, decision, launcher, runner=runner)
    write_record(replace(provisional, identity=identity))
    print(format_summary(plan, decision))
    return 0


def run_exec(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    hosts = _resolve_exec_hosts(args)
    host = hosts[0]
    command = normalize_remote_command(_strip_remainder(args.exec_command))
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
    provenance = _provenance_for_plan(plan, args=args, env=env, stdin_body=stdin_body)
    provisional = _record_from_plan(
        plan,
        decision,
        provenance=provenance,
        identity={"pending_launch": True},
    )
    write_record(provisional)
    identity = launch_tmux(plan, decision, launcher, runner=runner)
    write_record(replace(provisional, identity=identity))
    print(format_summary(plan, decision))
    return 0


def _query_job_identity(
    host: HostSpec,
    session: str,
    window: str,
    *,
    expected_identity: dict[str, Any] | None = None,
    runner=subprocess.run,
    timeout_seconds: int = 8,
) -> dict[str, Any]:
    try:
        proc = runner(
            _ssh_python_argv(host.ssh_host, connect_timeout=timeout_seconds),
            input=job_control.build_identity_source(session, window, expected_identity),
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 3,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise JobIdentityUnreachable(f"timed out reading job identity on {host.name} after {exc.timeout}s") from exc
    returncode = int(getattr(proc, "returncode", 1))
    if returncode != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        error = detail or f"failed to read job identity on {host.name}"
        if returncode == 255:
            raise JobIdentityUnreachable(error)
        raise JobIdentityProbeError(error)
    try:
        payload = job_control.parse_identity_stdout(getattr(proc, "stdout", "") or "")
    except ValueError as exc:
        raise JobIdentityProbeError(str(exc)) from exc
    if not payload.get("ok"):
        raise JobIdentityProbeError(payload.get("error") or f"job identity probe failed on {host.name}")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise JobIdentityProbeError(f"job identity probe returned invalid identity on {host.name}")
    return identity


def _record_from_plan(
    plan,
    decision,
    *,
    provenance: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> RunRecord:
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
        created_at=utc_now(),
        provenance=provenance or {},
        identity=identity or {},
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


def _forward_text_stream(stream: Any, destination: Any, *, on_read_error: Any) -> None:
    destination_error: BaseException | None = None
    try:
        while True:
            chunk = stream.read(8192)
            if chunk == "":
                break
            if destination_error is None:
                try:
                    destination.write(chunk)
                    destination.flush()
                except BaseException as exc:
                    destination_error = exc
    except BaseException:
        on_read_error()
        raise
    if destination_error is not None:
        raise destination_error


def _terminate_and_wait(proc: Any) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=TAIL_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _ssh_python_argv(host: str, *, connect_timeout: int | None = None) -> list[str]:
    timeout = None if connect_timeout in (None, 0) else int(connect_timeout)
    return build_remote_python_argv(host, timeout_seconds=timeout)


def run_tail(args: argparse.Namespace, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    record = read_record(args.run_ref)
    ensure_knuckles_master(runner=runner)
    if args.live:
        proc = popener(
            _ssh_python_argv(record.ssh_host),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            _terminate_and_wait(proc)
            raise RuntimeError("failed to open remote tail pipes")
        started = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ucl-tail-stderr") as executor:
            stderr_forward = executor.submit(
                _forward_text_stream,
                proc.stderr,
                sys.stderr,
                on_read_error=proc.terminate,
            )
            try:
                proc.stdin.write(_tail_follow_source(record.log_path, int(args.lines)))
                proc.stdin.close()
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
                _terminate_and_wait(proc)
                try:
                    stderr_forward.result()
                except Exception:
                    pass
                return 130
            except BaseException:
                _terminate_and_wait(proc)
                try:
                    stderr_forward.result()
                except Exception:
                    pass
                raise
            returncode = int(proc.wait())
            stderr_forward.result()
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
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
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
                if rel.endswith(SUFFIXES):
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
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        raise RuntimeError(detail or "remote clean failed")
    payload = json.loads(_extract_between(getattr(proc, "stdout", "") or "", CLEAN_SENTINEL_BEGIN, CLEAN_SENTINEL_END, label="clean"))
    for path in payload.get("paths", []):
        print(path)
    return int(getattr(proc, "returncode", 0))


def _record_status(record: RunRecord, *, runner=subprocess.run, timeout_seconds: int = 8) -> dict[str, Any]:
    host = HostSpec(name=record.host, ssh_host=record.ssh_host)
    try:
        current_identity = _query_job_identity(
            host,
            record.session,
            record.window,
            expected_identity=record.identity,
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
    except JobIdentityUnreachable as exc:
        return {"status": "unreachable", "error": str(exc), "current_identity": {}}
    except Exception as exc:  # noqa: BLE001 - status should report per-job probe failures.
        return {"status": "probe_error", "error": str(exc), "current_identity": {}}
    status = job_control.classify_identity(record.identity, current_identity)
    if status == "exited_or_missing" and record.kind == "stage-setup":
        stage_id = (record.provenance or {}).get("stage_id")
        if isinstance(stage_id, str) and stage_id:
            try:
                stage_record = stage_registry.read_record(stage_id)
                payload = dict(
                    stage_tools.probe_stage_state(
                        host,
                        ready_state_path=stage_record.state_path,
                        failed_state_path=_failed_state_path(stage_record.state_path),
                        runner=runner,
                    )
                )
                _validate_stage_state(stage_record, payload)
                stage_registry.update_status(stage_id, "ready")
                status = "completed"
            except Exception as error:  # noqa: BLE001 - job status must stay per-record.
                if "stage failed" in str(error):
                    try:
                        stage_registry.update_status(stage_id, "failed")
                    except ValueError:
                        pass
                    return {"status": "failed", "error": str(error), "current_identity": current_identity}
                return {"status": status, "error": str(error), "current_identity": current_identity}
    return {
        "status": status,
        "error": "",
        "current_identity": current_identity,
    }


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
        "identity": record.identity,
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
            "identity",
            "current_identity",
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


def _stop_source(
    session: str,
    window: str,
    signal: str,
    *,
    expected_identity: dict[str, Any] | None = None,
    grace_seconds: float = 5.0,
) -> str:
    return job_control.build_stop_source(
        session,
        window,
        expected_identity or {},
        signal,
        grace_seconds,
    )


def run_stop(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    record: RunRecord | None = None
    result: dict[str, Any]
    try:
        if args.run_ref == "last" and not args.yes:
            raise RuntimeError(
                "refusing to stop 'last' without --yes; pass an explicit run id or use 'ucl stop last --yes'"
            )
        if not math.isfinite(args.grace_seconds) or args.grace_seconds < 0:
            raise ValueError("--grace-seconds must be finite and non-negative")
        record = read_record(args.run_ref)
        ensure_knuckles_master(runner=runner)
        proc = runner(
            _ssh_python_argv(record.ssh_host, connect_timeout=args.timeout_seconds),
            input=_stop_source(
                record.session,
                record.window,
                args.signal,
                expected_identity=record.identity,
                grace_seconds=args.grace_seconds,
            ),
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds + args.grace_seconds + 5,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = {
            "ok": False,
            "status": "unknown_after_timeout",
            "signal": args.signal,
            "timed_out": True,
            "wrapper_error": True,
            "error": f"timed out stopping {record.run_id if record else args.run_ref} after {exc.timeout}s",
        }
    except Exception as exc:  # noqa: BLE001 - --json must remain machine-readable on setup failures.
        result = {
            "ok": False,
            "status": "wrapper_error",
            "signal": args.signal,
            "timed_out": False,
            "wrapper_error": True,
            "error": str(exc),
        }
    else:
        returncode = int(getattr(proc, "returncode", 1))
        if returncode != 0:
            detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
            result = {
                "ok": False,
                "status": "wrapper_error",
                "signal": args.signal,
                "timed_out": False,
                "wrapper_error": True,
                "remote_returncode": returncode,
                "error": detail or f"remote stop wrapper failed for {record.run_id if record else args.run_ref}",
            }
        else:
            try:
                result = job_control.parse_stop_stdout(getattr(proc, "stdout", "") or "")
            except Exception as exc:  # noqa: BLE001 - protocol errors are structured CLI results.
                detail = _wrapper_stream_detail(
                    getattr(proc, "stdout", "") or "",
                    getattr(proc, "stderr", "") or "",
                )
                error = str(exc)
                if detail:
                    error = f"{error}: {detail}"
                result = {
                    "ok": False,
                    "status": "wrapper_error",
                    "signal": args.signal,
                    "timed_out": False,
                    "wrapper_error": True,
                    "error": error,
                }
            else:
                result.setdefault("timed_out", False)
                result.setdefault("wrapper_error", False)
    payload: dict[str, Any] = {
        "run_id": record.run_id if record else args.run_ref,
        "host": record.host if record else "",
        "session": record.session if record else "",
        "window": record.window if record else "",
        **result,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_id: {record.run_id if record else args.run_ref}")
        if record:
            print(f"target: {record.session}:{record.window}")
        print(f"status: {result.get('status', 'unknown')}")
        if result.get("signal_errors"):
            print(f"signal_errors: {json.dumps(result['signal_errors'], sort_keys=True)}")
        if result.get("survivors"):
            print(f"survivors: {json.dumps(result['survivors'], sort_keys=True)}")
        if result.get("cleanup_error"):
            print(f"cleanup_error: {result['cleanup_error']}", file=sys.stderr)
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
    return 0 if result.get("ok") else 2


def _copy_endpoint_manifest(endpoint: copy_tools.Endpoint, *, verify: str, runner=subprocess.run) -> dict[str, Any]:
    return copy_tools.read_manifest(endpoint, sha256=(verify == "sha256"), runner=runner)


def _copy_transfer_argv(
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
    args: argparse.Namespace,
    *,
    selective: bool,
    source_is_directory: bool,
    dry_run: bool,
) -> tuple[str, list[str]]:
    if src.host and dst.host:
        mode = "remote-to-remote"
        builder = copy_tools.build_selective_remote_to_remote_argv if selective else copy_tools.build_remote_to_remote_argv
    else:
        mode = "rsync"
        builder = copy_tools.build_selective_rsync_argv if selective else copy_tools.build_rsync_argv
    kwargs: dict[str, Any] = {
        "partial": args.partial,
        "progress": args.progress,
        "dry_run": dry_run,
        "rsync_args": tuple(args.rsync_args),
    }
    if selective:
        kwargs["source_is_directory"] = source_is_directory
    return mode, builder(src, dst, **kwargs)


def _render_copy_payload(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if payload.get("stdout"):
        print(payload["stdout"], end="")
    if payload.get("stderr"):
        print(payload["stderr"], file=sys.stderr, end="")
    if payload.get("error"):
        print(f"error: {payload['error']}", file=sys.stderr)
    plan = payload.get("plan")
    if plan:
        print(
            "reconcile: "
            f"exact={len(plan['destination_exact'])} "
            f"reused={len(plan['reused'])} "
            f"transfer={len(plan['transfer_paths'])} "
            f"bytes={plan['bytes_to_transfer']}"
        )
    verify = payload.get("verify", {})
    if verify.get("mode") != "none":
        print(f"verify: {verify.get('message', 'unknown')}")


def _run_plain_copy(
    args: argparse.Namespace,
    *,
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
    runner=subprocess.run,
) -> int:
    mode, argv = _copy_transfer_argv(
        src,
        dst,
        args,
        selective=False,
        source_is_directory=False,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        payload = {
            "ok": True,
            "src": args.src,
            "dst": args.dst,
            "resolved_src": src.rsync_spec(),
            "resolved_dst": dst.rsync_spec(),
            "mode": mode,
            "argv": argv,
            "dry_run": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "plan": None,
            "attempts": [],
            "verify": {"mode": "none", "ok": None, "message": "skipped"},
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(" ".join(shlex.quote(part) for part in argv))
        return 0
    proc = runner(argv, capture_output=True, text=True, shell=False)
    ok = int(getattr(proc, "returncode", 1)) == 0
    payload = {
        "ok": ok,
        "src": args.src,
        "dst": args.dst,
        "resolved_src": src.rsync_spec(),
        "resolved_dst": dst.rsync_spec(),
        "mode": mode,
        "argv": argv,
        "returncode": int(getattr(proc, "returncode", 1)),
        # The framed rsync transport has already removed only pre-handshake
        # startup output. Everything after that boundary is rsync output.
        "stdout": getattr(proc, "stdout", "") or "",
        "stderr": getattr(proc, "stderr", "") or "",
        "dry_run": False,
        "plan": None,
        "attempts": [],
        "verify": {"mode": "none", "ok": None, "message": "skipped"},
    }
    _render_copy_payload(payload, json_output=args.json)
    return 0 if ok else 2


def _read_reconcile_manifests(
    endpoints: dict[str, copy_tools.Endpoint],
    *,
    verify: str,
    runner=subprocess.run,
) -> dict[str, dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = {
            name: executor.submit(_copy_endpoint_manifest, endpoint, verify=verify, runner=runner)
            for name, endpoint in endpoints.items()
        }
        return {name: future.result() for name, future in futures.items()}


def _verify_message(
    source: dict[str, Any],
    destination: dict[str, Any],
    diff: copy_tools.ManifestDiff,
    *,
    sha256: bool,
) -> str:
    if diff.ok:
        return "ok"
    _, message = copy_tools.compare_manifests(source, destination, sha256=sha256)
    if message == "ok":
        return "sha256 manifest differs" if sha256 else "size/path manifest differs"
    return message


def _render_reconciled_copy_error(
    args: argparse.Namespace,
    *,
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
    mode: str,
    plan: dict[str, Any],
    attempts: list[dict[str, Any]],
    error: Exception,
) -> int:
    payload = {
        "ok": False,
        "src": args.src,
        "dst": args.dst,
        "resolved_src": src.rsync_spec(),
        "resolved_dst": dst.rsync_spec(),
        "reuse_from": args.reuse_from,
        "mode": mode,
        "argv": attempts[-1]["argv"] if attempts else [],
        "returncode": 2,
        "stdout": "".join(str(attempt["stdout"]) for attempt in attempts),
        "stderr": "".join(str(attempt["stderr"]) for attempt in attempts),
        "error": str(error),
        "plan": plan,
        "attempts": attempts,
        "verify": {
            "mode": args.verify,
            "ok": False,
            "message": "post-transfer verification failed",
        },
    }
    _render_copy_payload(payload, json_output=args.json)
    return 2


def _run_reconciled_copy(
    args: argparse.Namespace,
    *,
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
    reuse: copy_tools.Endpoint | None,
    runner=subprocess.run,
) -> int:
    if args.retries < 0:
        raise ValueError("--retries must be a non-negative integer")
    if args.rsync_args:
        raise ValueError("raw rsync arguments are not supported with --verify; remove --verify or the raw arguments")
    if reuse is not None:
        if args.verify != "sha256":
            raise ValueError("--reuse-from requires --verify sha256")
        copy_tools.validate_reuse_endpoints(dst, reuse)

    if not args.json:
        labels = "source and destination" if reuse is None else "source, destination, and reuse tree"
        print(f"preflight: hashing {labels} with {args.verify}")
    endpoints = {"source": src, "destination": dst}
    if reuse is not None:
        endpoints["reuse"] = reuse
    manifests = _read_reconcile_manifests(endpoints, verify=args.verify, runner=runner)
    source_manifest = manifests["source"]
    destination_manifest = manifests["destination"]
    for label, manifest in manifests.items():
        unsupported = copy_tools.unsupported_entries(manifest)
        if unsupported:
            detail = ", ".join(f"{item['path']} ({item['kind']})" for item in unsupported[:5])
            raise ValueError(f"verified copy does not support {label} symlinks or special files: {detail}")
    if not source_manifest.get("exists"):
        raise ValueError(f"copy source does not exist: {args.src}")
    source_kind = source_manifest.get("root_kind", "directory")
    if source_kind not in {"file", "directory"}:
        raise ValueError(f"unsupported copy source type: {source_kind}")
    source_is_directory = source_kind == "directory"
    copy_tools.validate_reconcile_paths(src, dst, source_is_directory=source_is_directory)
    empty_directories = list(source_manifest.get("empty_directories", []))
    if empty_directories:
        preview = ", ".join(empty_directories[:5])
        raise ValueError(
            "verified copy does not support empty source directories because they cannot be content-verified "
            f"(examples: {preview})"
        )
    if source_is_directory and destination_manifest.get("root_kind") == "file":
        raise ValueError("verified directory copy destination cannot be an existing file")

    initial_diff = copy_tools.endpoint_diff(
        source_manifest,
        destination_manifest,
        source_endpoint=src,
        sha256=(args.verify == "sha256"),
    )
    initial_diff = copy_tools.ignore_destination_internal_partials(
        initial_diff,
        enabled=bool(args.partial and source_is_directory),
    )
    if source_is_directory and initial_diff.extra:
        preview = ", ".join(initial_diff.extra[:5])
        raise ValueError(
            "verified directory destination contains files absent from the source; "
            f"use a clean destination (examples: {preview})"
        )
    reusable: tuple[str, ...] = ()
    if reuse is not None:
        if not source_is_directory or manifests["reuse"].get("root_kind", "directory") != "directory":
            raise ValueError("--reuse-from currently requires directory source, reuse, and destination roots")
        reuse_diff = copy_tools.diff_manifests(source_manifest, manifests["reuse"], sha256=True)
        reusable = copy_tools.hardlinkable_paths(
            source_manifest,
            manifests["reuse"],
            tuple(sorted(set(initial_diff.transfer_paths).intersection(reuse_diff.exact))),
        )
    transfer_paths = tuple(sorted(set(initial_diff.transfer_paths) - set(reusable)))
    plan = {
        "comparison": args.verify,
        "source_files": int(source_manifest.get("file_count", 0)),
        "destination_exact": list(initial_diff.exact),
        "destination_missing": list(initial_diff.missing),
        "destination_mismatched": list(initial_diff.mismatched),
        "destination_extra": list(initial_diff.extra),
        "reuse_candidates": list(reusable),
        "reused": [],
        "transfer_paths": list(transfer_paths),
        "bytes_to_transfer": copy_tools.manifest_bytes(source_manifest, transfer_paths),
    }

    mode, planned_argv = _copy_transfer_argv(
        src,
        dst,
        args,
        selective=True,
        source_is_directory=source_is_directory,
        dry_run=True,
    )
    if args.dry_run:
        payload = {
            "ok": True,
            "src": args.src,
            "dst": args.dst,
            "resolved_src": src.rsync_spec(),
            "resolved_dst": dst.rsync_spec(),
            "reuse_from": args.reuse_from,
            "mode": mode,
            "argv": planned_argv if transfer_paths else [],
            "dry_run": True,
            "plan": plan,
            "verify": {"mode": args.verify, "ok": None, "message": "planned only"},
            "attempts": [],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }
        _render_copy_payload(payload, json_output=args.json)
        return 0

    if reusable and reuse is not None:
        plan["reused"] = copy_tools.hardlink_reusable(reuse, dst, reusable, runner=runner)

    attempts: list[dict[str, Any]] = []
    pending = transfer_paths
    final_manifest = destination_manifest
    final_diff = initial_diff
    if reusable:
        final_manifest = _copy_endpoint_manifest(dst, verify=args.verify, runner=runner)
        final_diff = copy_tools.endpoint_diff(
            source_manifest,
            final_manifest,
            source_endpoint=src,
            sha256=(args.verify == "sha256"),
        )
        final_diff = copy_tools.ignore_destination_internal_partials(
            final_diff,
            enabled=bool(args.partial and source_is_directory),
        )
        pending = final_diff.transfer_paths

    for attempt_number in range(1, args.retries + 2):
        if not pending:
            break
        mode, argv = _copy_transfer_argv(
            src,
            dst,
            args,
            selective=True,
            source_is_directory=source_is_directory,
            dry_run=False,
        )
        input_data = copy_tools.files_from_input(pending) if source_is_directory else None
        proc = runner(argv, input=input_data, capture_output=True, text=True, shell=False)
        attempt = {
            "attempt": attempt_number,
            "paths": list(pending),
            "argv": argv,
            "returncode": int(getattr(proc, "returncode", 1)),
            "stdout": getattr(proc, "stdout", "") or "",
            "stderr": getattr(proc, "stderr", "") or "",
            "remaining": list(pending),
        }
        attempts.append(attempt)
        try:
            final_manifest = _copy_endpoint_manifest(dst, verify=args.verify, runner=runner)
        except Exception as exc:  # Preserve the completed transfer's diagnostics.
            attempt["verification_error"] = str(exc)
            return _render_reconciled_copy_error(
                args,
                src=src,
                dst=dst,
                mode=mode,
                plan=plan,
                attempts=attempts,
                error=exc,
            )
        final_diff = copy_tools.endpoint_diff(
            source_manifest,
            final_manifest,
            source_endpoint=src,
            sha256=(args.verify == "sha256"),
        )
        final_diff = copy_tools.ignore_destination_internal_partials(
            final_diff,
            enabled=bool(args.partial and source_is_directory),
        )
        attempt["remaining"] = list(final_diff.transfer_paths)
        pending = final_diff.transfer_paths
        if not pending:
            break

    try:
        final_source_snapshot = _copy_endpoint_manifest(src, verify=args.verify, runner=runner)
    except Exception as exc:
        return _render_reconciled_copy_error(
            args,
            src=src,
            dst=dst,
            mode=mode,
            plan=plan,
            attempts=attempts,
            error=exc,
        )
    source_stable = copy_tools.source_snapshot_stable(
        source_manifest,
        final_source_snapshot,
        sha256=(args.verify == "sha256"),
    )
    verify_ok = final_diff.ok and source_stable
    transfer_ok = not attempts or attempts[-1]["returncode"] == 0
    ok = verify_ok and transfer_ok
    message = _verify_message(
        source_manifest,
        final_manifest,
        final_diff,
        sha256=(args.verify == "sha256"),
    )
    if not source_stable:
        message = "source changed during copy; result is not trusted"
    payload = {
        "ok": ok,
        "src": args.src,
        "dst": args.dst,
        "resolved_src": src.rsync_spec(),
        "resolved_dst": dst.rsync_spec(),
        "reuse_from": args.reuse_from,
        "mode": mode,
        "argv": attempts[-1]["argv"] if attempts else [],
        "returncode": attempts[-1]["returncode"] if attempts else 0,
        "stdout": "".join(str(attempt["stdout"]) for attempt in attempts),
        "stderr": "".join(str(attempt["stderr"]) for attempt in attempts),
        "plan": plan,
        "attempts": attempts,
        "verify": {
            "mode": args.verify,
            "ok": verify_ok,
            "message": message,
            "source_stable": source_stable,
            **final_diff.as_dict(),
        },
    }
    _render_copy_payload(payload, json_output=args.json)
    return 0 if ok else 2


def run_copy(args: argparse.Namespace, *, runner=subprocess.run) -> int:
    src = copy_tools.resolve_endpoint(copy_tools.parse_endpoint(args.src), args.catalog)
    dst = copy_tools.resolve_endpoint(copy_tools.parse_endpoint(args.dst), args.catalog)
    reuse = None
    if args.reuse_from:
        reuse = copy_tools.resolve_endpoint(copy_tools.parse_endpoint(args.reuse_from), args.catalog)
    needs_ssh = bool(src.host or dst.host or (reuse and reuse.host))
    if needs_ssh and (not args.dry_run or args.verify != "none"):
        ensure_knuckles_master(runner=runner)
    if args.verify == "none":
        if reuse is not None:
            raise ValueError("--reuse-from requires --verify sha256")
        return _run_plain_copy(args, src=src, dst=dst, runner=runner)
    return _run_reconciled_copy(args, src=src, dst=dst, reuse=reuse, runner=runner)


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
                "stdout": getattr(proc, "stdout", "") or "",
                "stderr": getattr(proc, "stderr", "") or "",
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
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "src": args.src,
                            "dst": args.dst,
                            "dry_run": bool(args.dry_run),
                            "returncode": 2,
                            "stdout": "",
                            "stderr": "",
                            "error": str(exc),
                            "plan": None,
                            "attempts": [],
                            "verify": {"mode": args.verify, "ok": False, "message": "not completed"},
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
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
        if args.command == "stage":
            return run_stage(args, runner=runner)
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
