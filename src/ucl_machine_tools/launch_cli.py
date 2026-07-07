"""CLI for launching a local script bundle on a UCL machine."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ucl_machine_tools.hosts import load_catalog, parse_selector
from ucl_machine_tools.launch import (
    build_plan,
    decide_tmux,
    format_summary,
    launch_tmux,
    list_remote_sessions,
    upload_bundle,
    write_launcher,
)
from ucl_machine_tools.ssh import ensure_knuckles_master


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a local script bundle and launch it in tmux on a UCL host.")
    parser.add_argument("--host", required=True, help="UCL host alias from the catalog")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--script", required=True, help="script path inside local-dir")
    parser.add_argument("--catalog", type=Path, help="host catalog JSON")
    parser.add_argument("--session", help="tmux session to use or create")
    parser.add_argument("--new-session", action="store_true", help="force creating a new tmux session")
    parser.add_argument("--window", help="tmux window name when launching into an existing session")
    parser.add_argument("--remote-dir", help="remote bundle dir under /tmp/ucl-machine-tools/launchers")
    parser.add_argument("--log", help="remote log path; default is <remote-dir>/run.log")
    parser.add_argument("--arg", action="append", default=[], help="script argument; repeat for multiple args")
    parser.add_argument("--env", action="append", default=[], help="remote env KEY=VALUE; repeat for multiple vars")
    parser.add_argument("--replace", action="store_true", help="replace an existing non-empty remote bundle dir")
    parser.add_argument("--dry-run", action="store_true", help="print planned actions without remote mutation")
    return parser


def _resolve_host(selector: str, *, catalog_path: Path | None):
    catalog = load_catalog(catalog_path)
    hosts = parse_selector(selector, catalog=catalog)
    if len(hosts) != 1:
        raise ValueError(f"--host must resolve to exactly one host, got {len(hosts)} for {selector!r}")
    return hosts[0]


def _dry_run_summary(plan) -> str:  # noqa: ANN001 - small CLI formatting helper.
    return "\n".join(
        [
            "dry_run: true",
            f"host:       {plan.host.name}",
            f"local_dir:  {plan.local_dir}",
            f"script:     {plan.script_rel}",
            f"remote_dir: {plan.remote_dir}",
            f"log:        {plan.log_path}",
            f"session:    {plan.requested_session or plan.generated_session}",
            f"window:     {plan.window}",
            "actions:",
            "  ensure knuckles SSH master",
            "  upload local dir with tar over SSH",
            "  write .ucl_launch.sh",
            "  discover tmux sessions",
            "  launch in tmux",
        ]
    )


def main(argv: list[str] | None = None, *, runner=subprocess.run, popener=subprocess.Popen) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        host = _resolve_host(args.host, catalog_path=args.catalog)
        plan = build_plan(
            host=host,
            local_dir=args.local_dir,
            script=args.script,
            session=args.session,
            remote_dir=args.remote_dir,
            log_path=args.log,
            window=args.window,
            args=args.arg,
            env=args.env,
            replace=args.replace,
            new_session=args.new_session,
        )
        if args.dry_run:
            print(_dry_run_summary(plan))
            return 0

        ensure_knuckles_master(runner=runner)
        sessions = list_remote_sessions(plan.host, runner=runner)
        decision = decide_tmux(
            sessions=sessions,
            generated_session=plan.generated_session,
            requested_session=plan.requested_session,
            new_session=plan.new_session,
            window=plan.window,
        )
        upload_bundle(plan, runner=runner, popener=popener)
        write_launcher(plan, runner=runner)
        launch_tmux(plan, decision, runner=runner)
        print(format_summary(plan, decision))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should render concise errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
