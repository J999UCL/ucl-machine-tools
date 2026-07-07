"""Local run registry for remote UCL jobs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    kind: str
    host: str
    ssh_host: str
    session: str
    window: str
    remote_dir: str
    log_path: str
    command: tuple[str, ...]


def registry_root() -> Path:
    base = os.environ.get("UCL_MACHINE_TOOLS_CACHE")
    if base:
        return Path(base).expanduser() / "runs"
    return Path("~/.cache/ucl-machine-tools/runs").expanduser()


def _record_path(run_id: str, *, root: Path | None = None) -> Path:
    return (root or registry_root()) / f"{run_id}.json"


def write_record(record: RunRecord, *, root: Path | None = None) -> None:
    target_root = root or registry_root()
    target_root.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["command"] = list(record.command)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _record_path(record.run_id, root=target_root).write_text(text, encoding="utf-8")
    (target_root / "latest.json").write_text(text, encoding="utf-8")


def read_record(ref: str = "last", *, root: Path | None = None) -> RunRecord:
    target_root = root or registry_root()
    path = target_root / "latest.json" if ref == "last" else _record_path(ref, root=target_root)
    if not path.exists():
        raise ValueError(f"run record not found: {ref}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["command"] = tuple(payload.get("command") or ())
    return RunRecord(**payload)
