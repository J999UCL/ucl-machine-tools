"""Rsync copy and explicit transfer verification helpers."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ucl_machine_tools.hosts import HostSpec, load_catalog
from ucl_machine_tools.ssh import build_remote_python_argv


Runner = Callable[..., subprocess.CompletedProcess]
MANIFEST_BEGIN = "UCL_COPY_MANIFEST_BEGIN"
MANIFEST_END = "UCL_COPY_MANIFEST_END"
SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Endpoint:
    raw: str
    host: str | None
    path: str

    @property
    def is_remote(self) -> bool:
        return self.host is not None

    def rsync_spec(self) -> str:
        if self.host is None:
            return self.path
        return f"{self.host}:{self.path}"


def parse_endpoint(value: str) -> Endpoint:
    if not value:
        raise ValueError("copy endpoint must be non-empty")
    if ":" in value and not value.startswith("/"):
        host, path = value.split(":", 1)
        if not SAFE_HOST_RE.match(host):
            raise ValueError(f"unsafe remote host in endpoint: {value!r}")
        if not path.startswith("/"):
            raise ValueError(f"remote endpoint path must be absolute: {value!r}")
        return Endpoint(raw=value, host=host, path=path)
    return Endpoint(raw=value, host=None, path=str(Path(value).expanduser()))


def resolve_endpoint_host(endpoint: Endpoint, catalog_path: Path | None = None) -> HostSpec | None:
    if endpoint.host is None:
        return None
    catalog = load_catalog(catalog_path)
    for spec in catalog.values():
        if endpoint.host in {spec.name, spec.ssh_host, *spec.aliases}:
            return spec
    raise ValueError(f"unknown remote copy host: {endpoint.host}")


def build_rsync_argv(src: Endpoint, dst: Endpoint, *, partial: bool = False, dry_run: bool = False) -> list[str]:
    argv = ["rsync", "-a", "--human-readable"]
    if partial:
        argv += ["--partial", "--append-verify"]
    if dry_run:
        argv.append("--dry-run")
    argv += ["-e", "ssh -o BatchMode=yes -o LogLevel=ERROR", src.rsync_spec(), dst.rsync_spec()]
    return argv


def build_remote_to_remote_argv(src: Endpoint, dst: Endpoint, *, partial: bool = False, dry_run: bool = False) -> list[str]:
    if src.host is None or dst.host is None:
        raise ValueError("remote-to-remote rsync requires two remote endpoints")
    rsync = build_rsync_argv(Endpoint(src.path, None, src.path), dst, partial=partial, dry_run=dry_run)
    command = " ".join(shlex.quote(part) for part in rsync)
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", src.host, "bash", "-lc", command]


def manifest_source(path: str, *, sha256: bool) -> str:
    return f"""
import hashlib
import json
import os
BEGIN={json.dumps(MANIFEST_BEGIN)}
END={json.dumps(MANIFEST_END)}
ROOT={json.dumps(path)}
SHA256={bool(sha256)!r}

def relpath(path):
    if os.path.isdir(ROOT):
        return os.path.relpath(path, ROOT)
    return os.path.basename(path)

files = []
total_bytes = 0
exists = os.path.exists(ROOT)
if exists:
    if os.path.isfile(ROOT):
        candidates = [ROOT]
    else:
        candidates = []
        for base, _, names in os.walk(ROOT):
            for name in names:
                candidates.append(os.path.join(base, name))
    for path in sorted(candidates):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        total_bytes += size
        item = {{"path": relpath(path), "bytes": size}}
        if SHA256:
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            item["sha256"] = h.hexdigest()
        files.append(item)
payload = {{"schema_version": 1, "exists": exists, "file_count": len(files), "total_bytes": total_bytes, "files": files}}
print(BEGIN)
print(json.dumps(payload, sort_keys=True))
print(END)
"""


def _extract_manifest(stdout: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != MANIFEST_BEGIN:
            continue
        for end_idx in range(idx + 1, len(lines)):
            if lines[end_idx].strip() == MANIFEST_END:
                return json.loads("\n".join(lines[idx + 1 : end_idx]))
    raise RuntimeError("copy manifest sentinel not found")


def endpoint_manifest(endpoint: Endpoint, *, sha256: bool, runner: Runner = subprocess.run) -> dict[str, Any]:
    if endpoint.host is None:
        return local_manifest(endpoint, sha256=sha256)
    proc = runner(
        build_remote_python_argv(endpoint.host),
        input=manifest_source(endpoint.path, sha256=sha256),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        raise RuntimeError(detail or f"remote manifest failed on {endpoint.host}")
    return _extract_manifest(getattr(proc, "stdout", "") or "")


def local_manifest(endpoint: Endpoint, *, sha256: bool) -> dict[str, Any]:
    if endpoint.host is not None:
        raise ValueError("local_manifest requires a local endpoint")
    root = Path(endpoint.path)
    files: list[dict[str, Any]] = []
    total = 0
    if root.exists():
        candidates = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        for path in candidates:
            size = path.stat().st_size
            total += size
            rel = path.name if root.is_file() else path.relative_to(root).as_posix()
            item: dict[str, Any] = {"path": rel, "bytes": size}
            if sha256:
                import hashlib

                h = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
                item["sha256"] = h.hexdigest()
            files.append(item)
    return {"schema_version": 1, "exists": root.exists(), "file_count": len(files), "total_bytes": total, "files": files}


def read_manifest(endpoint: Endpoint, *, sha256: bool, runner: Runner = subprocess.run) -> dict[str, Any]:
    if endpoint.host is None:
        return local_manifest(endpoint, sha256=sha256)
    return endpoint_manifest(endpoint, sha256=sha256, runner=runner)


def compare_manifests(before: dict[str, Any], after: dict[str, Any], *, sha256: bool) -> tuple[bool, str]:
    if before.get("file_count") != after.get("file_count"):
        return False, f"file_count differs: {before.get('file_count')} != {after.get('file_count')}"
    if before.get("total_bytes") != after.get("total_bytes"):
        return False, f"total_bytes differs: {before.get('total_bytes')} != {after.get('total_bytes')}"
    if sha256:
        left = {item["path"]: item.get("sha256") for item in before.get("files", [])}
        right = {item["path"]: item.get("sha256") for item in after.get("files", [])}
        if left != right:
            return False, "sha256 manifest differs"
    return True, "ok"
