#!/usr/bin/env python3
"""Read-only UCL GPU and /tmp/ucl-machine-tools state inventory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from ucl_machine_tools.hosts import load_catalog, parse_selector
from ucl_machine_tools.inventory import collect, format_table, to_jsonable
from ucl_machine_tools.ssh import ensure_knuckles_master


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check UCL GPU availability and /tmp/ucl-machine-tools state. "
            "Targets can be hosts, labels/classes such as 3090ti, groups such as timeshare, or all."
        )
    )
    parser.add_argument("command", nargs="?", choices=("check", "gpus", "state", "recommend"), default="check")
    parser.add_argument("target", nargs="?", help="host, comma list, group, GPU class, or all")
    parser.add_argument("--selector", help="explicit selector; overrides positional target")
    parser.add_argument("--catalog", type=Path, help="host catalog JSON")
    parser.add_argument("--root", default="/tmp/ucl-machine-tools", help="remote scratch root to check")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--table", action="store_true", help="emit a human table even when scripting")
    parser.add_argument("--jobs", type=int, default=4, help="parallel host probes")
    parser.add_argument("--timeout-seconds", type=int, default=8, help="SSH connect/probe timeout")
    parser.add_argument("--only-free", action="store_true", help="show only ready/free hosts")
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    parser.add_argument("--min-tmp-free-gb", type=float)
    parser.add_argument("--sizes", action="store_true", help="run a bounded du check for the scratch root")
    parser.add_argument("--debug", action="store_true", help="include parser/debug tails in JSON rows")
    return parser


def _selector_from_args(args: argparse.Namespace) -> str:
    if args.selector:
        return args.selector
    if args.target:
        return args.target
    return "all"


def _filter_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = rows
    if args.command == "gpus":
        out = [row for row in out if row.get("gpus")]
    if args.command == "state":
        out = [row for row in out if row.get("scratch") or row.get("status") not in {"no-gpu", "busy", "ready"}]
    if args.only_free or args.command == "recommend":
        out = [row for row in out if row.get("status") == "ready"]
    if args.command == "recommend":
        out = sorted(
            out,
            key=lambda row: (
                _best_free_vram_mb(row),
                _tmp_free_gb(row),
            ),
            reverse=True,
        )
        out = out[:1]
    return out


def _best_free_vram_mb(row: dict[str, Any]) -> float:
    best = 0.0
    for gpu in row.get("gpus", []) or []:
        free = gpu.get("memory_free_mb")
        if free is None and gpu.get("memory_total_mb") is not None and gpu.get("memory_used_mb") is not None:
            free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        best = max(best, float(free or 0))
    return best


def _tmp_free_gb(row: dict[str, Any]) -> float:
    for fs in row.get("filesystems", []) or []:
        if fs.get("path") == "/tmp":
            return float(fs.get("available_gb") or 0)
    return 0.0


def main(argv: list[str] | None = None, *, runner=subprocess.run) -> int:
    args = build_parser().parse_args(argv)
    catalog = load_catalog(args.catalog)
    root = args.root
    min_tmp_free_gb = float(args.min_tmp_free_gb) if args.min_tmp_free_gb is not None else 50.0
    selected = parse_selector(_selector_from_args(args), catalog=catalog)
    ensure_knuckles_master(runner=runner)
    rows = collect(
        selected,
        runner=runner,
        jobs=args.jobs,
        timeout_seconds=args.timeout_seconds,
        root=root,
        sizes=args.sizes,
        debug=args.debug,
        min_tmp_free_gb=min_tmp_free_gb,
        min_free_vram_gb=float(args.min_free_vram_gb),
    )
    rows = _filter_rows(args, rows)
    if args.json and not args.table:
        print(json.dumps(to_jsonable(rows), indent=2, sort_keys=True))
    else:
        print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
