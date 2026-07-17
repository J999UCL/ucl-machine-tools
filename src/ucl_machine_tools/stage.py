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
PATHS_SENTINEL_BEGIN = "UCL_STAGE_PATHS_JSON_BEGIN"
PATHS_SENTINEL_END = "UCL_STAGE_PATHS_JSON_END"
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


def build_source_probe_source(
    source_dir: str,
    manifest: SourceManifest | None = None,
    *,
    source_sha256: str | None = None,
) -> str:
    source = _safe_absolute_path(source_dir, "source_dir")
    marker = posixpath.join(source, SOURCE_MARKER_NAME)
    if (manifest is None) == (source_sha256 is None):
        raise ValueError("source probe requires exactly one of manifest or source_sha256")
    expected = manifest.as_dict() if manifest is not None else None
    expected_hash = manifest.source_sha256 if manifest is not None else source_sha256
    expected_literal = (
        f"json.loads({json.dumps(json.dumps(expected, sort_keys=True))})"
        if expected is not None
        else "None"
    )
    return f'''import hashlib
import json
import os
from pathlib import Path
import stat

BEGIN = {SOURCE_SENTINEL_BEGIN!r}
END = {SOURCE_SENTINEL_END!r}
SOURCE = Path({source!r})
MARKER = Path({marker!r})
EXPECTED_LITERAL = {expected_literal}
EXPECTED_HASH = {expected_hash!r}
payload = {{"schema_version": 1, "ok": True, "ready": False, "exists": SOURCE.exists(), "error": ""}}
if SOURCE.exists():
    try:
        marker = json.loads(MARKER.read_text(encoding="utf-8"))
        EXPECTED = EXPECTED_LITERAL if EXPECTED_LITERAL is not None else marker
        if marker != EXPECTED or marker.get("source_sha256") != EXPECTED_HASH or not SOURCE.is_dir() or SOURCE.is_symlink():
            raise ValueError("integrity marker or source directory does not match")
        marker_digest = hashlib.sha256()
        marker_digest.update(b"ucl-source-manifest-v1\\n")
        for entry in marker.get("entries", []):
            marker_digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
            marker_digest.update(b"\\n")
        if marker_digest.hexdigest() != EXPECTED_HASH:
            raise ValueError("integrity marker entries do not match the registered source identity")
        if SOURCE.stat().st_mode & 0o222 or MARKER.stat().st_mode & 0o222:
            raise ValueError("source snapshot is not immutable")
        actual = {{}}
        for base, directories, names in os.walk(SOURCE, followlinks=False):
            base_path = Path(base)
            for name in list(directories):
                path = base_path / name
                if path.is_symlink():
                    target = os.readlink(path)
                    actual[path.relative_to(SOURCE).as_posix()] = ("symlink", target, False, len(os.fsencode(target)))
                    directories.remove(name)
                elif path.stat().st_mode & 0o222:
                    raise ValueError(f"source directory is writable: {{path.relative_to(SOURCE).as_posix()}}")
                else:
                    info = path.stat()
                    actual[path.relative_to(SOURCE).as_posix()] = (
                        "directory",
                        None,
                        bool(stat.S_IMODE(info.st_mode) & 0o111),
                        0,
                    )
            for name in names:
                if base_path == SOURCE and name == {SOURCE_MARKER_NAME!r}:
                    continue
                path = base_path / name
                relative = path.relative_to(SOURCE).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    target = os.readlink(path)
                    actual[relative] = ("symlink", target, False, len(os.fsencode(target)))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_mode & 0o222:
                        raise ValueError(f"source file is writable: {{relative}}")
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    actual[relative] = ("file", digest.hexdigest(), bool(stat.S_IMODE(info.st_mode) & 0o111), info.st_size)
                else:
                    raise ValueError(f"unsupported source entry: {{relative}}")
        if set(actual) != {{entry["path"] for entry in EXPECTED["entries"]}}:
            raise ValueError("source paths differ from the staged manifest")
        for entry in EXPECTED["entries"]:
            kind, identity, executable, size = actual[entry["path"]]
            expected_identity = entry.get("sha256") if kind == "file" else entry.get("symlink_target")
            if (kind, identity, executable, size) != (entry["kind"], expected_identity, entry["executable"], entry["size"]):
                raise ValueError(f"source entry differs from the staged manifest: {{entry['path']}}")
        payload["ready"] = True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        payload["ok"] = False
        payload["error"] = f"existing source integrity verification failed: {{exc}}"
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


def build_managed_paths_probe_source(remote_root: str, managed_paths: tuple[str, ...]) -> str:
    """Build a read-only probe that rejects symlinked managed path components."""

    root = _safe_absolute_path(remote_root, "remote_root")
    normalized: list[str] = []
    for raw in managed_paths:
        path = _safe_absolute_path(raw, "managed_path")
        if posixpath.commonpath((root, path)) != root:
            raise ValueError(f"managed_path must be beneath remote_root: {raw!r}")
        if path not in normalized:
            normalized.append(path)
    return f'''import json
from pathlib import Path

BEGIN = {PATHS_SENTINEL_BEGIN!r}
END = {PATHS_SENTINEL_END!r}
ROOT = Path({root!r})
PATHS = tuple(Path(value) for value in {tuple(normalized)!r})
payload = {{"schema_version": 1, "ok": True, "error": ""}}
try:
    for path in (ROOT, *PATHS):
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ValueError(f"managed path component is a symlink: {{current}}")
except (OSError, ValueError) as exc:
    payload["ok"] = False
    payload["error"] = str(exc)
print(BEGIN)
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
print(END)
'''


def verify_managed_paths(
    host: HostSpec,
    *,
    remote_root: str,
    managed_paths: tuple[str, ...],
    runner: Runner = subprocess.run,
) -> None:
    payload = _run_remote_python(
        host,
        build_managed_paths_probe_source(remote_root, managed_paths),
        runner=runner,
        label="managed path preflight",
        begin=PATHS_SENTINEL_BEGIN,
        end=PATHS_SENTINEL_END,
    )
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "managed path preflight failed"))


def probe_remote_source(
    host: HostSpec,
    *,
    source_dir: str,
    manifest: SourceManifest,
    runner: Runner = subprocess.run,
) -> bool:
    payload = _run_remote_python(
        host,
        build_source_probe_source(source_dir, manifest),
        runner=runner,
        label="source probe",
        begin=SOURCE_SENTINEL_BEGIN,
        end=SOURCE_SENTINEL_END,
    )
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "remote source probe failed"))
    return bool(payload.get("ready"))


def verify_registered_source(
    host: HostSpec,
    *,
    source_dir: str,
    source_sha256: str,
    runner: Runner = subprocess.run,
) -> None:
    payload = _run_remote_python(
        host,
        build_source_probe_source(source_dir, source_sha256=source_sha256),
        runner=runner,
        label="registered source verification",
        begin=SOURCE_SENTINEL_BEGIN,
        end=SOURCE_SENTINEL_END,
    )
    if not payload.get("ok") or not payload.get("ready"):
        raise RuntimeError(str(payload.get("error") or "registered source is not ready"))


def verify_stage_environment(
    host: HostSpec,
    *,
    source_dir: str,
    environment_dir: str,
    uv_binary_path: str,
    uv_cache_dir: str,
    python_install_dir: str,
    python_request: str,
    runner: Runner = subprocess.run,
) -> None:
    source = _safe_absolute_path(source_dir, "source_dir")
    environment = _safe_absolute_path(environment_dir, "environment_dir")
    uv_binary = _safe_absolute_path(uv_binary_path, "uv_binary_path")
    uv_cache = _safe_absolute_path(uv_cache_dir, "uv_cache_dir")
    python_install = _safe_absolute_path(python_install_dir, "python_install_dir")
    check = runner(
        build_remote_argv(
            host.ssh_host,
            (
                "env",
                f"UV_PROJECT_ENVIRONMENT={environment}",
                f"UV_CACHE_DIR={uv_cache}",
                f"UV_PYTHON_INSTALL_DIR={python_install}",
                uv_binary,
                "sync",
                "--frozen",
                "--check",
                "--project",
                source,
            ),
        ),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(check, "returncode", 1)) != 0:
        returncode = int(getattr(check, "returncode", 1))
        detail = (getattr(check, "stderr", "") or getattr(check, "stdout", "") or "").strip()
        detail = detail or f"remote check exited {returncode} without output"
        raise RuntimeError(f"staged environment consistency check failed on {host.name}: {detail}")

    python_path = posixpath.join(environment, "bin", "python")
    version_source = (
        "import platform,sys; request=sys.argv[1]; implementation,sep,version=request.rpartition('-'); "
        "implementation=implementation if sep else ''; version=version if sep else request; "
        "raise SystemExit(0 if platform.python_version()==version and "
        "(not implementation or sys.implementation.name==implementation) else 1)"
    )
    version = runner(
        build_remote_argv(host.ssh_host, (python_path, "-c", version_source, python_request)),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(version, "returncode", 1)) != 0:
        returncode = int(getattr(version, "returncode", 1))
        detail = (getattr(version, "stderr", "") or getattr(version, "stdout", "") or "").strip()
        detail = detail or f"remote check exited {returncode} without output"
        raise RuntimeError(f"staged Python identity check failed on {host.name}: {detail}")


def build_source_prepare_source(
    *,
    incoming_dir: str,
    sources_dir: str,
    directories: tuple[str, ...] = (),
) -> str:
    incoming = _safe_absolute_path(incoming_dir, "incoming_dir")
    sources = _safe_absolute_path(sources_dir, "sources_dir")
    if posixpath.commonpath((incoming, sources)) != sources or incoming == sources:
        raise ValueError("incoming_dir must be beneath sources_dir")
    for directory in directories:
        path = PurePosixPath(directory)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError(f"unsafe source directory path: {directory!r}")
    return f'''from pathlib import Path
import shutil

incoming = Path({incoming!r})
sources = Path({sources!r})
for candidate in (sources, *sources.parents):
    if candidate != Path("/") and candidate.is_symlink():
        raise SystemExit(f"managed source path component is a symlink: {{candidate}}")
sources.mkdir(parents=True, exist_ok=True)
if incoming.exists() or incoming.is_symlink():
    incoming.unlink() if incoming.is_symlink() else shutil.rmtree(incoming)
incoming.mkdir(mode=0o700)
for relative in {directories!r}:
    (incoming / relative).mkdir(parents=True, exist_ok=True)
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

def inventory(root, *, marker_name=None, require_immutable=False):
    if not root.is_dir() or root.is_symlink():
        fail(f"source snapshot is missing, redirected, or not a directory: {{root}}")
    if require_immutable and root.stat().st_mode & 0o222:
        fail(f"source snapshot root is writable: {{root}}")
    actual = {{}}
    for base, directories, names in os.walk(root, followlinks=False):
        base_path = Path(base)
        for name in list(directories):
            path = base_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                target = os.readlink(path)
                actual[relative] = ("symlink", target, False, len(os.fsencode(target)))
                directories.remove(name)
            else:
                info = path.stat()
                if require_immutable and info.st_mode & 0o222:
                    fail(f"source directory is writable: {{path.relative_to(root).as_posix()}}")
                actual[path.relative_to(root).as_posix()] = (
                    "directory",
                    None,
                    bool(stat.S_IMODE(info.st_mode) & 0o111),
                    0,
                )
        for name in names:
            if marker_name is not None and base_path == root and name == marker_name:
                continue
            path = base_path / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                actual[relative] = ("symlink", target, False, len(os.fsencode(target)))
            elif stat.S_ISREG(info.st_mode):
                if require_immutable and info.st_mode & 0o222:
                    fail(f"source file is writable: {{relative}}")
                actual[relative] = ("file", digest(path), bool(stat.S_IMODE(info.st_mode) & 0o111), info.st_size)
            else:
                fail(f"unsupported staged source entry: {{relative}}")
    return actual

def verify_snapshot(root, *, marker_name=None, require_immutable=False):
    actual = inventory(root, marker_name=marker_name, require_immutable=require_immutable)
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

try:
    verify_snapshot(INCOMING)

    marker = EXPECTED
    marker_path = INCOMING / MARKER_NAME
    marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\\n", encoding="utf-8")
    marker_path.chmod(0o600)

    SOURCES.mkdir(parents=True, exist_ok=True)
    with (SOURCES / ".ucl-source.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reused = False
        if FINAL.is_symlink():
            fail("existing final source path must not be a symlink")
        if FINAL.exists():
            try:
                existing = json.loads((FINAL / MARKER_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"existing final source has invalid integrity marker: {{exc}}")
            if existing != EXPECTED:
                fail("existing final source has a different integrity marker")
            existing_digest = hashlib.sha256()
            existing_digest.update(b"ucl-source-manifest-v1\\n")
            for entry in existing.get("entries", []):
                existing_digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
                existing_digest.update(b"\\n")
            if existing_digest.hexdigest() != EXPECTED["source_sha256"]:
                fail("existing final source has a forged or corrupt integrity marker")
            verify_snapshot(FINAL, marker_name=MARKER_NAME, require_immutable=True)
            shutil.rmtree(INCOMING)
            reused = True
        else:
            expected_by_path = {{entry["path"]: entry for entry in EXPECTED["entries"]}}
            marker_path.chmod(0o444)
            for base, directories, names in os.walk(INCOMING, topdown=False, followlinks=False):
                base_path = Path(base)
                for name in names:
                    path = base_path / name
                    if path.is_symlink() or path == marker_path:
                        continue
                    entry = expected_by_path[path.relative_to(INCOMING).as_posix()]
                    path.chmod(0o555 if entry["executable"] else 0o444)
                for name in directories:
                    path = base_path / name
                    if not path.is_symlink():
                        path.chmod(0o555)
            os.replace(INCOMING, FINAL)
            FINAL.chmod(0o555)
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
        manifest=manifest,
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
        input=build_source_prepare_source(
            incoming_dir=incoming,
            sources_dir=sources_dir,
            directories=tuple(entry.path for entry in manifest.entries if entry.kind == "directory"),
        ),
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
    paths = [entry.path for entry in manifest.entries if entry.kind != "directory"]
    transfer = runner(
        argv,
        input=copy_tools.files_from_input(paths),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(transfer, "returncode", 1)) != 0:
        detail = (getattr(transfer, "stderr", "") or getattr(transfer, "stdout", "") or "").strip()
        cleanup = runner(
            build_remote_argv(
                host.ssh_host,
                (
                    "python3",
                    "-c",
                    "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
                    "shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)",
                    incoming,
                ),
            ),
            capture_output=True,
            text=True,
            shell=False,
        )
        if int(getattr(cleanup, "returncode", 1)) != 0:
            cleanup_detail = (getattr(cleanup, "stderr", "") or getattr(cleanup, "stdout", "") or "").strip()
            detail = f"{detail or 'source upload failed'}; cleanup failed: {cleanup_detail or 'unknown error'}"
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
candidates = [path for path in (ready, failed) if path.is_file() and not path.is_symlink()]
target = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path == failed)) if candidates else None
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
                if path is not None and field != "python_path" and path.is_symlink():
                    exists = False
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
