"""Local run registry for remote UCL jobs."""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    created_at: str = ""
    updated_at: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def registry_root() -> Path:
    base = os.environ.get("UCL_MACHINE_TOOLS_CACHE")
    if base:
        return Path(base).expanduser() / "runs"
    return Path("~/.cache/ucl-machine-tools/runs").expanduser()


def _record_path(run_id: str, *, root: Path | None = None) -> Path:
    return (root or registry_root()) / f"{run_id}.json"


def _atomic_write(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_record(record: RunRecord, *, root: Path | None = None) -> None:
    target_root = root or registry_root()
    target_root.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    identity = payload.pop("identity")
    provenance = payload["provenance"]
    provenance["job_identity"] = identity
    now = utc_now()
    payload["created_at"] = payload.get("created_at") or now
    payload["updated_at"] = now
    payload["command"] = list(record.command)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    lock_path = target_root / ".registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        record_path = _record_path(record.run_id, root=target_root)
        _atomic_write(record_path, text)
        _atomic_write(target_root / "latest.json", text)


def read_record(ref: str = "last", *, root: Path | None = None) -> RunRecord:
    target_root = root or registry_root()
    path = target_root / "latest.json" if ref == "last" else _record_path(ref, root=target_root)
    if not path.exists():
        raise ValueError(f"run record not found: {ref}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if ref == "last" and "kind" not in payload and isinstance(payload.get("run_id"), str):
        path = _record_path(payload["run_id"], root=target_root)
        if not path.exists():
            raise ValueError(f"latest run record target not found: {payload['run_id']}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["command"] = tuple(payload.get("command") or ())
    payload.setdefault("created_at", "")
    payload.setdefault("updated_at", "")
    payload.setdefault("provenance", {})
    legacy_identity = payload.pop("identity", None)
    payload["identity"] = (
        legacy_identity
        if legacy_identity is not None
        else payload["provenance"].get("job_identity", {})
    )
    return RunRecord(**payload)


def list_records(*, root: Path | None = None) -> list[RunRecord]:
    target_root = root or registry_root()
    if not target_root.exists():
        return []
    records: list[RunRecord] = []
    for path in sorted(target_root.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            records.append(read_record(path.stem, root=target_root))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda record: record.updated_at or record.created_at or record.run_id)
