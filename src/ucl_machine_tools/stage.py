"""Remote orchestration helpers for content-addressed UV stages."""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Mapping

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.ssh import build_remote_argv, build_remote_python_argv, describe_ssh_failure
from ucl_machine_tools.uv_project import SourceManifest
from ucl_machine_tools.uv_remote import UvSetupPayload


Runner = Callable[..., subprocess.CompletedProcess]
SOURCE_SENTINEL_BEGIN = "UCL_STAGE_SOURCE_JSON_BEGIN"
SOURCE_SENTINEL_END = "UCL_STAGE_SOURCE_JSON_END"
STATE_SENTINEL_BEGIN = "UCL_STAGE_STATE_JSON_BEGIN"
STATE_SENTINEL_END = "UCL_STAGE_STATE_JSON_END"
SOURCE_MARKER_NAME = ".ucl-stage-source.json"


@dataclass(frozen=True)
class SourceSyncResult:
    source_dir: str
    source_sha256: str
    reused: bool
    file_count: int
    total_bytes: int


def _safe_absolute_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(f"{label} must be a safe absolute path: {value!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} contains control characters")
    return posixpath.normpath(value)


def _parse_sentinel(stdout: str, begin: str, end: str, label: str) -> dict[str, object]:
    lines = stdout.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == begin]
    finishes = [index for index, line in enumerate(lines) if line.strip() == end]
    if len(starts) != 1 or len(finishes) != 1 or finishes[0] <= starts[0]:
        raise RuntimeError(f"{label} returned an invalid sentinel result")
    payload_lines = lines[starts[0] + 1 : finishes[0]]
    if len(payload_lines) != 1:
        raise RuntimeError(f"{label} sentinel payload must be one JSON line")
    try:
        payload = json.loads(payload_lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"{label} returned an unsupported result")
    return payload


def build_source_probe_source(source_dir: str, source_sha256: str) -> str:
    source = _safe_absolute_path(source_dir, "source_dir")
    marker = posixpath.join(source, SOURCE_MARKER_NAME)
    return f'''import json
from pathlib import Path

BEGIN = {SOURCE_SENTINEL_BEGIN!r}
END = {SOURCE_SENTINEL_END!r}
SOURCE = Path({source!r})
MARKER = Path({marker!r})
EXPECTED = {source_sha256!r}
payload = {{"schema_version": 1, "ok": True, "ready": False, "exists": SOURCE.exists(), "error": ""}}
if SOURCE.exists():
    try:
        marker = json.loads(MARKER.read_text(encoding="utf-8"))
        payload["ready"] = marker.get("schema_version") == 1 and marker.get("source_sha256") == EXPECTED
        if not payload["ready"]:
            payload["ok"] = False
            payload["error"] = "existing source directory has no matching integrity marker"
    except (OSError, json.JSONDecodeError) as exc:
        payload["ok"] = False
        payload["error"] = f"existing source integrity marker is unreadable: {{exc}}"
print(BEGIN)
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
print(END)
'''


def _run_remote_python(
    host: HostSpec,
    source: str,
    *,
    runner: Runner,
    label: str,
    begin: str,
    end: str,
) -> dict[str, object]:
    process = runner(
        build_remote_python_argv(host.ssh_host, timeout_seconds=30),
        input=source,
        capture_output=True,
        text=True,
        shell=False,
    )
    stdout = getattr(process, "stdout", "") or ""
    if int(getattr(process, "returncode", 1)) != 0:
        try:
            payload = _parse_sentinel(stdout, begin, end, label)
        except RuntimeError:
            detail = describe_ssh_failure(
                int(getattr(process, "returncode", 1)),
                stdout=stdout,
                stderr=getattr(process, "stderr", "") or "",
            )
            raise RuntimeError(f"{label} failed on {host.name}: {detail}")
        raise RuntimeError(str(payload.get("error") or f"{label} failed on {host.name}"))
    return _parse_sentinel(stdout, begin, end, label)


def probe_remote_source(
    host: HostSpec,
    *,
    source_dir: str,
    source_sha256: str,
    runner: Runner = subprocess.run,
) -> bool:
    payload = _run_remote_python(
        host,
        build_source_probe_source(source_dir, source_sha256),
        runner=runner,
        label="source probe",
        begin=SOURCE_SENTINEL_BEGIN,
        end=SOURCE_SENTINEL_END,
    )
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "remote source probe failed"))
    return bool(payload.get("ready"))


def build_source_prepare_source(*, incoming_dir: str, sources_dir: str) -> str:
    incoming = _safe_absolute_path(incoming_dir, "incoming_dir")
    sources = _safe_absolute_path(sources_dir, "sources_dir")
    if posixpath.commonpath((incoming, sources)) != sources or incoming == sources:
        raise ValueError("incoming_dir must be beneath sources_dir")
    return f'''from pathlib import Path
import shutil

incoming = Path({incoming!r})
sources = Path({sources!r})
sources.mkdir(parents=True, exist_ok=True)
if incoming.exists() or incoming.is_symlink():
    shutil.rmtree(incoming)
incoming.mkdir(mode=0o700)
'''


def build_source_promote_source(
    *,
    manifest: SourceManifest,
    incoming_dir: str,
    source_dir: str,
    sources_dir: str,
) -> str:
    incoming = _safe_absolute_path(incoming_dir, "incoming_dir")
    source = _safe_absolute_path(source_dir, "source_dir")
    sources = _safe_absolute_path(sources_dir, "sources_dir")
    for value, label in ((incoming, "incoming_dir"), (source, "source_dir")):
        if posixpath.commonpath((value, sources)) != sources or value == sources:
            raise ValueError(f"{label} must be beneath sources_dir")
    expected = manifest.as_dict()
    return f'''import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

BEGIN = {SOURCE_SENTINEL_BEGIN!r}
END = {SOURCE_SENTINEL_END!r}
INCOMING = Path({incoming!r})
FINAL = Path({source!r})
SOURCES = Path({sources!r})
EXPECTED = json.loads({json.dumps(json.dumps(expected, sort_keys=True))})
MARKER_NAME = {SOURCE_MARKER_NAME!r}

def fail(message):
    if INCOMING.exists() and INCOMING != FINAL:
        shutil.rmtree(INCOMING, ignore_errors=True)
    payload = {{"schema_version": 1, "ok": False, "error": str(message)}}
    print(BEGIN)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    print(END)
    raise SystemExit(1)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

try:
    if not INCOMING.is_dir() or INCOMING.is_symlink():
        fail("incoming source snapshot is missing or is not a directory")
    actual = {{}}
    for base, directories, names in os.walk(INCOMING, followlinks=False):
        base_path = Path(base)
        for name in list(directories):
            path = base_path / name
            if path.is_symlink():
                relative = path.relative_to(INCOMING).as_posix()
                actual[relative] = ("symlink", os.readlink(path), False, len(os.fsencode(os.readlink(path))))
                directories.remove(name)
        for name in names:
            path = base_path / name
            relative = path.relative_to(INCOMING).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                actual[relative] = ("symlink", target, False, len(os.fsencode(target)))
            elif stat.S_ISREG(info.st_mode):
                actual[relative] = ("file", digest(path), bool(stat.S_IMODE(info.st_mode) & 0o111), info.st_size)
            else:
                fail(f"unsupported staged source entry: {{relative}}")

    expected_paths = {{entry["path"] for entry in EXPECTED["entries"]}}
    if set(actual) != expected_paths:
        missing = sorted(expected_paths - set(actual))
        extra = sorted(set(actual) - expected_paths)
        fail(f"source snapshot paths differ; missing={{missing[:5]}} extra={{extra[:5]}}")
    for entry in EXPECTED["entries"]:
        kind, identity, executable, size = actual[entry["path"]]
        if kind != entry["kind"] or size != entry["size"] or executable != entry["executable"]:
            fail(f"source metadata mismatch: {{entry['path']}}")
        expected_identity = entry.get("sha256") if kind == "file" else entry.get("symlink_target")
        if identity != expected_identity:
            fail(f"source content mismatch: {{entry['path']}}")

    marker = {{
        "schema_version": 1,
        "source_sha256": EXPECTED["source_sha256"],
        "file_count": EXPECTED["file_count"],
        "symlink_count": EXPECTED["symlink_count"],
        "total_bytes": EXPECTED["total_bytes"],
    }}
    marker_path = INCOMING / MARKER_NAME
    marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\\n", encoding="utf-8")
    marker_path.chmod(0o600)

    SOURCES.mkdir(parents=True, exist_ok=True)
    with (SOURCES / ".ucl-source.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reused = False
        if FINAL.exists():
            try:
                existing = json.loads((FINAL / MARKER_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"existing final source has invalid integrity marker: {{exc}}")
            if existing.get("source_sha256") != EXPECTED["source_sha256"]:
                fail("existing final source has a different source identity")
            shutil.rmtree(INCOMING)
            reused = True
        else:
            os.replace(INCOMING, FINAL)
    payload = {{
        "schema_version": 1,
        "ok": True,
        "reused": reused,
        "source_dir": str(FINAL),
        "source_sha256": EXPECTED["source_sha256"],
        "file_count": EXPECTED["file_count"],
        "total_bytes": EXPECTED["total_bytes"],
    }}
    print(BEGIN)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    print(END)
except SystemExit:
    raise
except Exception as exc:
    fail(f"source verification failed: {{type(exc).__name__}}: {{exc}}")
'''


def sync_source_snapshot(
    host: HostSpec,
    *,
    manifest: SourceManifest,
    source_dir: str,
    sources_dir: str,
    runner: Runner = subprocess.run,
) -> SourceSyncResult:
    if probe_remote_source(
        host,
        source_dir=source_dir,
        source_sha256=manifest.source_sha256,
        runner=runner,
    ):
        return SourceSyncResult(
            source_dir=source_dir,
            source_sha256=manifest.source_sha256,
            reused=True,
            file_count=sum(entry.kind == "file" for entry in manifest.entries),
            total_bytes=manifest.total_bytes,
        )

    incoming = posixpath.join(sources_dir, f".incoming-{manifest.source_sha256[:16]}-{uuid.uuid4().hex}")
    prepare = runner(
        build_remote_python_argv(host.ssh_host, timeout_seconds=30),
        input=build_source_prepare_source(incoming_dir=incoming, sources_dir=sources_dir),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(prepare, "returncode", 1)) != 0:
        detail = describe_ssh_failure(
            int(getattr(prepare, "returncode", 1)),
            stdout=getattr(prepare, "stdout", "") or "",
            stderr=getattr(prepare, "stderr", "") or "",
        )
        raise RuntimeError(f"could not prepare source upload on {host.name}: {detail}")

    source_endpoint = copy_tools.Endpoint(str(manifest.root), None, str(manifest.root))
    destination_endpoint = copy_tools.Endpoint(
        f"{host.ssh_host}:{incoming}", host.ssh_host, incoming
    )
    argv = copy_tools.build_selective_rsync_argv(
        source_endpoint,
        destination_endpoint,
        source_is_directory=True,
    )
    paths = [entry.path for entry in manifest.entries]
    transfer = runner(
        argv,
        input=copy_tools.files_from_input(paths),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(transfer, "returncode", 1)) != 0:
        detail = (getattr(transfer, "stderr", "") or getattr(transfer, "stdout", "") or "").strip()
        raise RuntimeError(detail or f"source upload failed on {host.name}")

    payload = _run_remote_python(
        host,
        build_source_promote_source(
            manifest=manifest,
            incoming_dir=incoming,
            source_dir=source_dir,
            sources_dir=sources_dir,
        ),
        runner=runner,
        label="source verification",
        begin=SOURCE_SENTINEL_BEGIN,
        end=SOURCE_SENTINEL_END,
    )
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "remote source verification failed"))
    return SourceSyncResult(
        source_dir=str(payload["source_dir"]),
        source_sha256=str(payload["source_sha256"]),
        reused=bool(payload.get("reused")),
        file_count=int(payload["file_count"]),
        total_bytes=int(payload["total_bytes"]),
    )


def write_setup_payload(
    host: HostSpec,
    payload: UvSetupPayload,
    *,
    runner: Runner = subprocess.run,
) -> None:
    for remote_path, source in payload.files.items():
        path = _safe_absolute_path(remote_path, "setup file path")
        script = 'set -euo pipefail; umask 077; cat > "$1"; chmod 700 "$1"'
        process = runner(
            build_remote_argv(host.ssh_host, ("bash", "--noprofile", "--norc", "-c", script, "bash", path)),
            input=source,
            capture_output=True,
            text=True,
            shell=False,
        )
        if int(getattr(process, "returncode", 1)) != 0:
            detail = describe_ssh_failure(
                int(getattr(process, "returncode", 1)),
                stdout=getattr(process, "stdout", "") or "",
                stderr=getattr(process, "stderr", "") or "",
            )
            raise RuntimeError(f"failed to write UV setup file on {host.name}: {detail}")


def build_state_probe_source(
    ready_state_path: str,
    failed_state_path: str,
    *,
    required_script: str | None = None,
) -> str:
    ready = _safe_absolute_path(ready_state_path, "ready_state_path")
    failed = _safe_absolute_path(failed_state_path, "failed_state_path")
    if required_script is not None:
        script = PurePosixPath(required_script)
        if script.is_absolute() or ".." in script.parts or str(script) in {"", "."}:
            raise ValueError(f"required_script must be a safe relative path: {required_script!r}")
    return f'''import json
from pathlib import Path

BEGIN = {STATE_SENTINEL_BEGIN!r}
END = {STATE_SENTINEL_END!r}
ready = Path({ready!r})
failed = Path({failed!r})
required_script = {required_script!r}
payload = {{"schema_version": 1, "status": "missing", "state": None}}
target = ready if ready.is_file() else failed if failed.is_file() else None
if target is not None:
    try:
        payload["state"] = json.loads(target.read_text(encoding="utf-8"))
        payload["status"] = "ready" if target == ready else "failed"
        state = payload["state"]
        checks = {{
            "source_dir": "dir",
            "environment_dir": "dir",
            "uv_binary_path": "file",
            "python_path": "file",
        }}
        missing = []
        if isinstance(state, dict):
            for field, kind in checks.items():
                value = state.get(field)
                path = Path(value) if isinstance(value, str) and value.startswith("/") else None
                exists = path is not None and (path.is_dir() if kind == "dir" else path.is_file())
                if not exists:
                    missing.append(field)
            if required_script:
                source_dir = state.get("source_dir")
                script_path = Path(source_dir) / required_script if isinstance(source_dir, str) else None
                if script_path is None or not script_path.is_file():
                    missing.append(f"script:{{required_script}}")
        payload["missing_paths"] = missing
    except (OSError, json.JSONDecodeError) as exc:
        payload["status"] = "invalid"
        payload["error"] = str(exc)
print(BEGIN)
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
print(END)
'''


def probe_stage_state(
    host: HostSpec,
    *,
    ready_state_path: str,
    failed_state_path: str,
    required_script: str | None = None,
    runner: Runner = subprocess.run,
) -> Mapping[str, object]:
    return _run_remote_python(
        host,
        build_state_probe_source(
            ready_state_path,
            failed_state_path,
            required_script=required_script,
        ),
        runner=runner,
        label="stage state probe",
        begin=STATE_SENTINEL_BEGIN,
        end=STATE_SENTINEL_END,
    )
