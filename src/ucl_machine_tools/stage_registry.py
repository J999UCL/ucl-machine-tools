"""Atomic local registry for reusable remote UV stages."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping


_STAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UV_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][A-Za-z0-9.-]+)?$")
_PYTHON_REQUEST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_STAGE_STATUSES = frozenset(
    {"unknown", "planned", "preparing", "ready", "failed", "launch_failed"}
)
_REQUIRED_FIELDS = (
    "stage_id",
    "name",
    "host",
    "ssh_host",
    "remote_root",
    "source_path",
    "environment_path",
    "uv_path",
    "cache_path",
    "source_hash",
    "lock_hash",
    "setup_environment_hash",
    "uv_version",
    "python_request",
)
_ADDITIVE_DEFAULTS: dict[str, Any] = {
    "python_path": "",
    "state_path": "",
    "setup_run_id": "",
    "status": "unknown",
    "created_at": "",
    "updated_at": "",
    "provenance": {},
}
_RECORD_FIELDS = frozenset((*_REQUIRED_FIELDS, *_ADDITIVE_DEFAULTS))
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "token",
    }
)


@dataclass(frozen=True)
class StageRecord:
    stage_id: str
    name: str
    host: str
    ssh_host: str
    remote_root: str
    source_path: str
    environment_path: str
    uv_path: str
    cache_path: str
    source_hash: str
    lock_hash: str
    setup_environment_hash: str
    uv_version: str
    python_request: str
    python_path: str = ""
    state_path: str = ""
    setup_run_id: str = ""
    status: str = "unknown"
    created_at: str = ""
    updated_at: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    """Return a stable UTC timestamp for persisted records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def registry_root() -> Path:
    """Return the local stage registry directory."""

    base = os.environ.get("UCL_MACHINE_TOOLS_CACHE")
    if base:
        return Path(base).expanduser() / "stages"
    return Path("~/.cache/ucl-machine-tools/stages").expanduser()


def _validate_stage_id(stage_id: str) -> None:
    if (
        not isinstance(stage_id, str)
        or stage_id == "last"
        or _STAGE_ID_PATTERN.fullmatch(stage_id) is None
    ):
        raise ValueError(f"invalid stage id: {stage_id!r}")


def _record_path(stage_id: str, *, root: Path) -> Path:
    _validate_stage_id(stage_id)
    return root / f"{stage_id}.json"


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


def _sensitive_key(key: str) -> bool:
    parts = {part for part in re.split(r"[^a-z0-9]+", key.lower()) if part}
    normalized = "".join(parts)
    return bool(parts & _SENSITIVE_KEY_PARTS) or normalized.endswith(
        ("apikey", "accesskey", "privatekey")
    )


def _sanitize_provenance(value: Any, *, redact_values: bool = False) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("stage provenance keys must be strings")
            if redact_values or _sensitive_key(raw_key):
                sanitized[raw_key] = "<redacted>"
            else:
                sanitized[raw_key] = _sanitize_provenance(
                    child,
                    redact_values=raw_key.lower() in {"env", "environment"},
                )
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_provenance(item, redact_values=redact_values) for item in value]
    if redact_values:
        return "<redacted>"
    return copy.deepcopy(value)


def _validate_record(record: StageRecord) -> None:
    _validate_stage_id(record.stage_id)
    for field_name in _REQUIRED_FIELDS:
        value = getattr(record, field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid stage record {record.stage_id}: {field_name} must be non-empty")
    for field_name in ("python_path", "state_path", "setup_run_id", "status", "created_at", "updated_at"):
        if not isinstance(getattr(record, field_name), str):
            raise ValueError(f"invalid stage record {record.stage_id}: {field_name} must be a string")
    if not isinstance(record.provenance, dict):
        raise ValueError(f"invalid stage record {record.stage_id}: provenance must be an object")
    for field_name in ("name", "host", "ssh_host"):
        value = getattr(record, field_name)
        if _SAFE_TOKEN_PATTERN.fullmatch(value) is None:
            raise ValueError(f"invalid stage record {record.stage_id}: unsafe {field_name}")
    for field_name in ("source_hash", "lock_hash", "setup_environment_hash"):
        if _SHA256_PATTERN.fullmatch(getattr(record, field_name)) is None:
            raise ValueError(f"invalid stage record {record.stage_id}: {field_name} must be SHA-256")
    if _UV_VERSION_PATTERN.fullmatch(record.uv_version) is None:
        raise ValueError(f"invalid stage record {record.stage_id}: invalid uv_version")
    if _PYTHON_REQUEST_PATTERN.fullmatch(record.python_request) is None:
        raise ValueError(f"invalid stage record {record.stage_id}: invalid python_request")
    if record.status not in _STAGE_STATUSES:
        raise ValueError(f"invalid stage record {record.stage_id}: invalid status {record.status!r}")
    if record.setup_run_id and _STAGE_ID_PATTERN.fullmatch(record.setup_run_id) is None:
        raise ValueError(f"invalid stage record {record.stage_id}: invalid setup_run_id")

    root = PurePosixPath(record.remote_root)
    if not record.remote_root.startswith("/") or root == PurePosixPath("/") or ".." in root.parts:
        raise ValueError(f"invalid stage record {record.stage_id}: remote_root must be a safe absolute path")
    for field_name in (
        "source_path",
        "environment_path",
        "uv_path",
        "cache_path",
        "python_path",
        "state_path",
    ):
        raw = getattr(record, field_name)
        if not raw:
            continue
        path = PurePosixPath(raw)
        if not raw.startswith("/") or ".." in path.parts or path == root:
            raise ValueError(f"invalid stage record {record.stage_id}: unsafe {field_name}")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"invalid stage record {record.stage_id}: {field_name} must be under remote_root"
            ) from error
    if PurePosixPath(record.source_path).name != record.source_hash:
        raise ValueError(f"invalid stage record {record.stage_id}: source_path does not match source_hash")
    expected_uv = root / "tools" / "uv" / record.uv_version / "uv"
    if PurePosixPath(record.uv_path) != expected_uv:
        raise ValueError(f"invalid stage record {record.stage_id}: uv_path does not match managed layout")
    if PurePosixPath(record.cache_path) != root / "cache" / "uv":
        raise ValueError(f"invalid stage record {record.stage_id}: cache_path does not match managed layout")


def _prepared_record(record: StageRecord, *, now: str | None = None) -> StageRecord:
    _validate_record(record)
    timestamp = now or utc_now()
    provenance = _sanitize_provenance(record.provenance)
    try:
        json.dumps(provenance)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid stage record {record.stage_id}: provenance is not JSON serializable") from error
    return replace(
        record,
        created_at=record.created_at or timestamp,
        updated_at=timestamp,
        provenance=provenance,
    )


def _record_text(record: StageRecord) -> str:
    return json.dumps(asdict(record), indent=2, sort_keys=True) + "\n"


def _write_unlocked(record: StageRecord, *, root: Path) -> None:
    _atomic_write(_record_path(record.stage_id, root=root), _record_text(record))
    pointer = json.dumps({"stage_id": record.stage_id}, indent=2, sort_keys=True) + "\n"
    _atomic_write(root / "latest.json", pointer)


def write_record(record: StageRecord, *, root: Path | None = None) -> StageRecord:
    """Atomically persist a stage record and update the latest pointer."""

    prepared = _prepared_record(record)
    target_root = root or registry_root()
    target_root.mkdir(parents=True, exist_ok=True)
    with (target_root / ".registry.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _write_unlocked(prepared, root=target_root)
    return prepared


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stage record JSON: {label}: {error.msg}") from error
    except OSError as error:
        raise ValueError(f"unable to read stage record {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid stage record {label}: expected an object")
    return payload


def _record_from_payload(payload: dict[str, Any], *, label: str) -> StageRecord:
    missing = [field_name for field_name in _REQUIRED_FIELDS if field_name not in payload]
    if missing:
        raise ValueError(
            f"invalid stage record {label}: missing required field(s): {', '.join(missing)}"
        )
    normalized = {key: value for key, value in payload.items() if key in _RECORD_FIELDS}
    for field_name, default in _ADDITIVE_DEFAULTS.items():
        normalized.setdefault(field_name, copy.deepcopy(default))
    try:
        record = StageRecord(**normalized)
    except TypeError as error:
        raise ValueError(f"invalid stage record {label}: {error}") from error
    _validate_record(record)
    sanitized = _sanitize_provenance(record.provenance)
    return replace(record, provenance=sanitized)


def _read_exact_unlocked(stage_id: str, *, root: Path, missing_message: str | None = None) -> StageRecord:
    path = _record_path(stage_id, root=root)
    if not path.exists():
        raise ValueError(missing_message or f"stage record not found: {stage_id}")
    return _record_from_payload(_load_json(path, label=stage_id), label=stage_id)


def _read_unlocked(ref: str, *, root: Path) -> StageRecord:
    if ref != "last":
        return _read_exact_unlocked(ref, root=root)

    pointer_path = root / "latest.json"
    if not pointer_path.exists():
        raise ValueError("stage record not found: last")
    pointer = _load_json(pointer_path, label="latest pointer")
    stage_id = pointer.get("stage_id")
    if not isinstance(stage_id, str):
        raise ValueError("invalid latest stage pointer: missing stage_id")
    _validate_stage_id(stage_id)
    return _read_exact_unlocked(
        stage_id,
        root=root,
        missing_message=f"latest stage record target not found: {stage_id}",
    )


def read_record(ref: str = "last", *, root: Path | None = None) -> StageRecord:
    """Read a stage by exact ID or the ``last`` pointer."""

    target_root = root or registry_root()
    if not target_root.exists():
        raise ValueError(f"stage record not found: {ref}")
    return _read_unlocked(ref, root=target_root)


def list_records(*, root: Path | None = None) -> list[StageRecord]:
    """List valid stage records in stable chronological order."""

    target_root = root or registry_root()
    if not target_root.exists():
        return []
    records: list[StageRecord] = []
    for path in sorted(target_root.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            records.append(_read_exact_unlocked(path.stem, root=target_root))
        except (OSError, TypeError, ValueError):
            continue
    return sorted(
        records,
        key=lambda record: (
            record.updated_at or record.created_at,
            record.created_at,
            record.stage_id,
        ),
    )


def update_status(
    ref: str,
    status: str,
    *,
    setup_run_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> StageRecord:
    """Immutably update stage status in one locked read-modify-write operation."""

    if not isinstance(status, str) or not status:
        raise ValueError("stage status must be a non-empty string")
    if setup_run_id is not None and not isinstance(setup_run_id, str):
        raise ValueError("setup_run_id must be a string")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("stage provenance update must be an object")

    target_root = root or registry_root()
    if not target_root.exists():
        raise ValueError(f"stage record not found: {ref}")
    with (target_root / ".registry.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _read_unlocked(ref, root=target_root)
        merged_provenance = copy.deepcopy(current.provenance)
        if provenance is not None:
            merged_provenance.update(copy.deepcopy(dict(provenance)))
        candidate = replace(
            current,
            status=status,
            setup_run_id=current.setup_run_id if setup_run_id is None else setup_run_id,
            provenance=merged_provenance,
        )
        prepared = _prepared_record(candidate)
        _write_unlocked(prepared, root=target_root)
    return prepared
