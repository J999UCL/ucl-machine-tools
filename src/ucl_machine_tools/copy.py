"""Rsync copy and explicit transfer verification helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ucl_machine_tools import rsync_transport
from ucl_machine_tools.hosts import HostSpec, load_catalog, parse_selector
from ucl_machine_tools.ssh import build_remote_argv, build_remote_python_argv


Runner = Callable[..., subprocess.CompletedProcess]
MANIFEST_BEGIN = "UCL_COPY_MANIFEST_BEGIN"
MANIFEST_END = "UCL_COPY_MANIFEST_END"
PRESENCE_BEGIN = "UCL_COPY_PRESENCE_BEGIN"
PRESENCE_END = "UCL_COPY_PRESENCE_END"
LINK_BEGIN = "UCL_COPY_LINK_BEGIN"
LINK_END = "UCL_COPY_LINK_END"
RSYNC_SSH = rsync_transport.build_transport_command()
SAFE_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_.!,:-]+$")
PARTIAL_DIR_NAME = ".ucl-rsync-partial"


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
            if self.path.startswith("-"):
                return f"./{self.path}"
            return self.path
        return f"{self.host}:{self.path}"


@dataclass(frozen=True)
class ManifestDiff:
    """A source-to-destination content comparison."""

    exact: tuple[str, ...]
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    extra: tuple[str, ...]

    @property
    def transfer_paths(self) -> tuple[str, ...]:
        return tuple(sorted((*self.missing, *self.mismatched)))

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched and not self.extra

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact": list(self.exact),
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
            "extra": list(self.extra),
            "transfer_paths": list(self.transfer_paths),
        }


def parse_endpoint(value: str) -> Endpoint:
    if not value:
        raise ValueError("copy endpoint must be non-empty")
    if ":" in value and not value.startswith("/"):
        host, path = value.split(":", 1)
        if not SAFE_SELECTOR_RE.match(host):
            raise ValueError(f"unsafe remote selector in endpoint: {value!r}")
        if not path.startswith("/"):
            raise ValueError(f"remote endpoint path must be absolute: {value!r}")
        return Endpoint(raw=value, host=host, path=path)
    expanded = str(Path(value).expanduser())
    if value.endswith("/") and expanded != "/":
        expanded = expanded.rstrip("/") + "/"
    return Endpoint(raw=value, host=None, path=expanded)


def resolve_endpoint_host(endpoint: Endpoint, catalog_path: Path | None = None) -> HostSpec | None:
    if endpoint.host is None:
        return None
    catalog = load_catalog(catalog_path)
    hosts = parse_selector(endpoint.host, catalog=catalog)
    if len(hosts) != 1:
        raise ValueError(f"copy endpoint selector must resolve to exactly one host, got {len(hosts)} for {endpoint.host!r}")
    return hosts[0]


def resolve_endpoint(endpoint: Endpoint, catalog_path: Path | None = None) -> Endpoint:
    host = resolve_endpoint_host(endpoint, catalog_path)
    if host is None:
        return endpoint
    return Endpoint(raw=endpoint.raw, host=host.ssh_host, path=endpoint.path)


def build_rsync_argv(
    src: Endpoint,
    dst: Endpoint,
    *,
    partial: bool = False,
    progress: bool = False,
    dry_run: bool = False,
    rsync_args: tuple[str, ...] = (),
) -> list[str]:
    validate_rsync_args(rsync_args)
    argv = ["rsync", "-a", "--human-readable", "-e", RSYNC_SSH]
    if partial:
        argv.append("--partial")
    if progress:
        argv.append("--info=progress2")
    if dry_run:
        argv.append("--dry-run")
    if src.is_remote or dst.is_remote:
        # This is the rsync 3.2.5-compatible spelling of secluded arguments.
        # The explicit option overrides RSYNC_OLD_ARGS/RSYNC_PROTECT_ARGS.
        argv.append("--protect-args")
    argv.extend(rsync_args)
    argv += ["--", src.rsync_spec(), dst.rsync_spec()]
    return argv


def build_remote_to_remote_argv(
    src: Endpoint,
    dst: Endpoint,
    *,
    partial: bool = False,
    progress: bool = False,
    dry_run: bool = False,
    rsync_args: tuple[str, ...] = (),
) -> list[str]:
    if src.host is None or dst.host is None:
        raise ValueError("remote-to-remote rsync requires two remote endpoints")
    rsync = build_rsync_argv(
        Endpoint(src.path, None, src.path),
        dst,
        partial=partial,
        progress=progress,
        dry_run=dry_run,
        rsync_args=rsync_args,
    )
    return rsync_transport.build_transport_argv(src.host, rsync, forward_agent=True)


def build_remote_destination_probe_argv(source_host: str, destination_host: str) -> list[str]:
    """Build a framed source-side SSH authentication probe."""

    return build_remote_argv(
        source_host,
        (
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=30",
            destination_host,
            "true",
        ),
        forward_agent=True,
    )


def validate_rsync_args(rsync_args: Sequence[str]) -> None:
    """Reject options that could replace or contaminate the framed transport."""

    for token in rsync_args:
        short_options = token[1:] if token.startswith("-") and not token.startswith("--") else ""
        is_short_rsh = _short_option_before_value(short_options, "e")
        is_short_remote_option = _short_option_before_value(short_options, "M")
        is_long_rsh = token == "--rsh" or token.startswith("--rsh=")
        is_remote_path = token == "--rsync-path" or token.startswith("--rsync-path=")
        is_long_remote_option = token == "--remote-option" or token.startswith("--remote-option=")
        disables_safe_args = token in {"--old-args", "--no-s", "--no-secluded-args", "--no-protect-args"}
        if (
            token == "--"
            or is_short_rsh
            or is_short_remote_option
            or is_long_rsh
            or is_remote_path
            or is_long_remote_option
            or disables_safe_args
        ):
            raise ValueError(f"raw rsync argument {token!r} cannot override ucl's framed transport")


def _short_option_before_value(options: str, target: str) -> bool:
    """Find a short option before an attached-value option consumes the suffix."""

    for option in options:
        if option == target:
            return True
        if option in {"B", "T", "f"}:
            return False
    return False


def manifest_source(path: str, *, sha256: bool) -> str:
    return f"""
import hashlib
import json
import os
import stat
BEGIN={json.dumps(MANIFEST_BEGIN)}
END={json.dumps(MANIFEST_END)}
ROOT={json.dumps(path)}
SHA256={bool(sha256)!r}

files = []
unsupported = []
empty_directories = []
total_bytes = 0
exists = os.path.lexists(ROOT)
root_kind = "missing"

def relative(path):
    return os.path.relpath(path, ROOT) if root_kind == "directory" else os.path.basename(path)

def inspect(path):
    global total_bytes
    info = os.lstat(path)
    rel = relative(path)
    if stat.S_ISLNK(info.st_mode):
        unsupported.append({{"path": rel, "kind": "symlink"}})
        return
    if not stat.S_ISREG(info.st_mode):
        unsupported.append({{"path": rel, "kind": "special"}})
        return
    total_bytes += info.st_size
    item = {{
        "path": rel,
        "bytes": info.st_size,
        "kind": "file",
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "inode": info.st_ino,
    }}
    if SHA256:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        item["sha256"] = h.hexdigest()
    files.append(item)

if exists:
    root_info = os.lstat(ROOT)
    if stat.S_ISREG(root_info.st_mode):
        root_kind = "file"
        inspect(ROOT)
    elif stat.S_ISDIR(root_info.st_mode):
        root_kind = "directory"
        for base, directories, names in os.walk(ROOT, followlinks=False):
            directories[:] = sorted(directories)
            for name in list(directories):
                candidate = os.path.join(base, name)
                if os.path.islink(candidate):
                    inspect(candidate)
                    directories.remove(name)
            for name in sorted(names):
                inspect(os.path.join(base, name))
            if not directories and not names:
                empty_directories.append(os.path.relpath(base, ROOT))
    elif stat.S_ISLNK(root_info.st_mode):
        root_kind = "symlink"
        unsupported.append({{"path": os.path.basename(ROOT), "kind": "symlink"}})
    else:
        root_kind = "special"
        unsupported.append({{"path": os.path.basename(ROOT), "kind": "special"}})
payload = {{
    "schema_version": 1,
    "exists": exists,
    "root_kind": root_kind,
    "file_count": len(files),
    "total_bytes": total_bytes,
    "files": sorted(files, key=lambda item: item["path"]),
    "unsupported": sorted(unsupported, key=lambda item: item["path"]),
    "empty_directories": sorted(empty_directories),
}}
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


def presence_source(path: str) -> str:
    """Return a tiny remote probe for the destination postcondition."""

    return f"""
import json, os
print({PRESENCE_BEGIN!r})
print(json.dumps({{"exists": os.path.lexists({path!r})}}, sort_keys=True))
print({PRESENCE_END!r})
"""


def endpoint_exists(endpoint: Endpoint, *, runner: Runner = subprocess.run) -> bool:
    if endpoint.host is None:
        return os.path.lexists(endpoint.path)
    proc = runner(
        build_remote_python_argv(endpoint.host),
        input=presence_source(endpoint.path),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        raise RuntimeError(detail or f"remote destination check failed on {endpoint.host}")
    payload = _extract_between(
        getattr(proc, "stdout", "") or "",
        PRESENCE_BEGIN,
        PRESENCE_END,
        "copy destination presence",
    )
    if not isinstance(payload.get("exists"), bool):
        raise RuntimeError("copy destination presence response did not contain a boolean exists field")
    return payload["exists"]


def local_manifest(endpoint: Endpoint, *, sha256: bool) -> dict[str, Any]:
    if endpoint.host is not None:
        raise ValueError("local_manifest requires a local endpoint")
    root = Path(endpoint.path)
    files: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    empty_directories: list[str] = []
    total = 0
    exists = os.path.lexists(root)
    root_kind = "missing"
    if exists:
        root_info = root.lstat()
        if stat.S_ISREG(root_info.st_mode):
            root_kind = "file"
            candidates = [root]
        elif stat.S_ISDIR(root_info.st_mode):
            root_kind = "directory"
            candidates = []
            for base, directories, names in os.walk(root, followlinks=False):
                directories[:] = sorted(directories)
                for name in list(directories):
                    candidate = Path(base) / name
                    if candidate.is_symlink():
                        unsupported.append({"path": candidate.relative_to(root).as_posix(), "kind": "symlink"})
                        directories.remove(name)
                candidates.extend(Path(base) / name for name in sorted(names))
                if not directories and not names:
                    empty_directories.append(Path(base).relative_to(root).as_posix() or ".")
        elif stat.S_ISLNK(root_info.st_mode):
            root_kind = "symlink"
            candidates = []
            unsupported.append({"path": root.name, "kind": "symlink"})
        else:
            root_kind = "special"
            candidates = []
            unsupported.append({"path": root.name, "kind": "special"})
        for path in candidates:
            info = path.lstat()
            rel = path.name if root_kind == "file" else path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                unsupported.append({"path": rel, "kind": "symlink"})
                continue
            if not stat.S_ISREG(info.st_mode):
                unsupported.append({"path": rel, "kind": "special"})
                continue
            total += info.st_size
            item: dict[str, Any] = {
                "path": rel,
                "bytes": info.st_size,
                "kind": "file",
                "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns,
                "uid": info.st_uid,
                "gid": info.st_gid,
                "inode": info.st_ino,
            }
            if sha256:
                h = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
                item["sha256"] = h.hexdigest()
            files.append(item)
    return {
        "schema_version": 1,
        "exists": exists,
        "root_kind": root_kind,
        "file_count": len(files),
        "total_bytes": total,
        "files": sorted(files, key=lambda item: item["path"]),
        "unsupported": sorted(unsupported, key=lambda item: item["path"]),
        "empty_directories": sorted(empty_directories),
    }


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


def _manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["path"]): item for item in manifest.get("files", [])}


def _items_match(left: dict[str, Any], right: dict[str, Any], *, sha256: bool) -> bool:
    if left.get("kind", "file") != right.get("kind", "file"):
        return False
    if left.get("bytes") != right.get("bytes"):
        return False
    if sha256:
        left_hash = left.get("sha256")
        right_hash = right.get("sha256")
        if not left_hash or left_hash != right_hash:
            return False
    return all(left.get(field) == right.get(field) for field in ("mode", "mtime_ns"))


def diff_manifests(source: dict[str, Any], destination: dict[str, Any], *, sha256: bool) -> ManifestDiff:
    """Compare directory manifests by relative path."""

    left = _manifest_index(source)
    right = _manifest_index(destination)
    exact: list[str] = []
    missing: list[str] = []
    mismatched: list[str] = []
    for path in sorted(left):
        if path not in right:
            missing.append(path)
        elif _items_match(left[path], right[path], sha256=sha256):
            exact.append(path)
        else:
            mismatched.append(path)
    return ManifestDiff(
        exact=tuple(exact),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
        extra=tuple(sorted(set(right) - set(left))),
    )


def endpoint_diff(
    source: dict[str, Any],
    destination: dict[str, Any],
    *,
    source_endpoint: Endpoint,
    sha256: bool,
) -> ManifestDiff:
    """Compare endpoint contents while retaining rsync's single-file semantics."""

    if source.get("root_kind") != "file":
        return diff_manifests(source, destination, sha256=sha256)

    source_files = list(source.get("files", []))
    if len(source_files) != 1:
        raise ValueError("file source manifest must contain exactly one file")
    source_item = source_files[0]
    destination_files = _manifest_index(destination)
    if destination.get("root_kind") == "file":
        candidates = list(destination.get("files", []))
        destination_item = candidates[0] if len(candidates) == 1 else None
    elif destination.get("root_kind") == "directory":
        destination_item = destination_files.get(Path(source_endpoint.path).name)
    else:
        destination_item = None
    label = str(source_item["path"])
    if destination_item is None:
        return ManifestDiff(exact=(), missing=(label,), mismatched=(), extra=())
    if _items_match(source_item, destination_item, sha256=sha256):
        return ManifestDiff(exact=(label,), missing=(), mismatched=(), extra=())
    return ManifestDiff(exact=(), missing=(), mismatched=(label,), extra=())


def manifest_bytes(manifest: dict[str, Any], paths: tuple[str, ...] | list[str]) -> int:
    index = _manifest_index(manifest)
    return sum(int(index[path].get("bytes", 0)) for path in paths if path in index)


def unsupported_entries(manifest: dict[str, Any]) -> tuple[dict[str, str], ...]:
    return tuple(manifest.get("unsupported", ()))


def hardlinkable_paths(
    source: dict[str, Any],
    reuse: dict[str, Any],
    candidates: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Return byte-identical files whose archive-preserved metadata also matches."""

    source_index = _manifest_index(source)
    reuse_index = _manifest_index(reuse)
    result: list[str] = []
    for path in candidates:
        left = source_index.get(path)
        right = reuse_index.get(path)
        if left is None or right is None or not _items_match(left, right, sha256=True):
            continue
        if all(left.get(field) == right.get(field) for field in ("mode", "mtime_ns", "uid", "gid")):
            result.append(path)
    return tuple(sorted(result))


def source_snapshot_stable(before: dict[str, Any], after: dict[str, Any], *, sha256: bool) -> bool:
    """Require the source tree to remain identical across reconciliation."""

    for field in ("exists", "root_kind", "unsupported", "empty_directories"):
        if before.get(field) != after.get(field):
            return False
    left = _manifest_index(before)
    right = _manifest_index(after)
    if set(left) != set(right):
        return False
    return all(
        _items_match(left[path], right[path], sha256=sha256)
        and all(left[path].get(field) == right[path].get(field) for field in ("uid", "gid", "inode"))
        for path in left
    )


def ignore_destination_internal_partials(diff: ManifestDiff, *, enabled: bool) -> ManifestDiff:
    if not enabled:
        return diff
    prefix = PARTIAL_DIR_NAME + "/"
    extras = tuple(path for path in diff.extra if path != PARTIAL_DIR_NAME and not path.startswith(prefix))
    return ManifestDiff(
        exact=diff.exact,
        missing=diff.missing,
        mismatched=diff.mismatched,
        extra=extras,
    )


def validate_reconcile_paths(source: Endpoint, destination: Endpoint, *, source_is_directory: bool) -> None:
    if not source_is_directory or source.host != destination.host:
        return
    source_path = Path(os.path.normpath(source.path))
    destination_path = Path(os.path.normpath(destination.path))
    if source_path == destination_path:
        return
    if source_path in destination_path.parents or destination_path in source_path.parents:
        raise ValueError("verified directory copy requires non-overlapping source and destination roots")


def _validated_relative_path(path: str) -> str:
    candidate = Path(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts or "\x00" in path:
        raise ValueError(f"unsafe manifest path: {path!r}")
    return candidate.as_posix()


def files_from_input(paths: tuple[str, ...] | list[str]) -> str:
    return "".join(f"{_validated_relative_path(path)}\x00" for path in paths)


def _directory_endpoint(endpoint: Endpoint) -> Endpoint:
    return Endpoint(raw=endpoint.raw, host=endpoint.host, path=endpoint.path.rstrip("/") + "/")


def build_selective_rsync_argv(
    src: Endpoint,
    dst: Endpoint,
    *,
    source_is_directory: bool,
    partial: bool = False,
    progress: bool = False,
    dry_run: bool = False,
    rsync_args: tuple[str, ...] = (),
) -> list[str]:
    if not source_is_directory:
        selective_args = ["--ignore-times"]
        if partial:
            selective_args.append(f"--partial-dir={PARTIAL_DIR_NAME}")
        return build_rsync_argv(
            src,
            dst,
            partial=False,
            progress=progress,
            dry_run=dry_run,
            rsync_args=(*selective_args, *rsync_args),
        )
    selective_args = ["--ignore-times", "--recursive", "--from0", "--files-from=-"]
    if partial:
        selective_args.append(f"--partial-dir={PARTIAL_DIR_NAME}")
    return build_rsync_argv(
        _directory_endpoint(src),
        _directory_endpoint(dst),
        partial=False,
        progress=progress,
        dry_run=dry_run,
        rsync_args=(*selective_args, *rsync_args),
    )


def build_selective_remote_to_remote_argv(
    src: Endpoint,
    dst: Endpoint,
    *,
    source_is_directory: bool,
    partial: bool = False,
    progress: bool = False,
    dry_run: bool = False,
    rsync_args: tuple[str, ...] = (),
) -> list[str]:
    if not source_is_directory:
        selective_args = ["--ignore-times"]
        if partial:
            selective_args.append(f"--partial-dir={PARTIAL_DIR_NAME}")
        return build_remote_to_remote_argv(
            src,
            dst,
            partial=False,
            progress=progress,
            dry_run=dry_run,
            rsync_args=(*selective_args, *rsync_args),
        )
    selective_args = ["--ignore-times", "--recursive", "--from0", "--files-from=-"]
    if partial:
        selective_args.append(f"--partial-dir={PARTIAL_DIR_NAME}")
    return build_remote_to_remote_argv(
        _directory_endpoint(src),
        _directory_endpoint(dst),
        partial=False,
        progress=progress,
        dry_run=dry_run,
        rsync_args=(*selective_args, *rsync_args),
    )


def validate_reuse_endpoints(destination: Endpoint, reuse: Endpoint) -> None:
    if destination.host != reuse.host:
        raise ValueError("--reuse-from must be local to the destination (both local or the same resolved host)")


def _atomic_hardlink(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"hard-link source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.ucl-link-{uuid.uuid4().hex}"
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def hardlink_local(reuse: Endpoint, destination: Endpoint, paths: tuple[str, ...] | list[str]) -> list[str]:
    if reuse.host is not None or destination.host is not None:
        raise ValueError("hardlink_local requires local endpoints")
    linked: list[str] = []
    for relative in paths:
        safe = _validated_relative_path(relative)
        _atomic_hardlink(Path(reuse.path) / safe, Path(destination.path) / safe)
        linked.append(safe)
    return linked


def hardlink_source(reuse_root: str, destination_root: str, paths: tuple[str, ...] | list[str]) -> str:
    validated = [_validated_relative_path(path) for path in paths]
    return f"""
import json
import os
import uuid
BEGIN={json.dumps(LINK_BEGIN)}
END={json.dumps(LINK_END)}
SOURCE={json.dumps(reuse_root)}
DESTINATION={json.dumps(destination_root)}
PATHS=json.loads({json.dumps(json.dumps(validated))})
linked=[]
for relative in PATHS:
    source=os.path.join(SOURCE, relative)
    destination=os.path.join(DESTINATION, relative)
    if not os.path.isfile(source) or os.path.islink(source):
        raise RuntimeError("hard-link source is not a regular file: " + source)
    parent=os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary=os.path.join(parent, "." + os.path.basename(destination) + ".ucl-link-" + uuid.uuid4().hex)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    linked.append(relative)
print(BEGIN)
print(json.dumps({{"schema_version": 1, "linked": linked}}, sort_keys=True))
print(END)
"""


def hardlink_remote(
    reuse: Endpoint,
    destination: Endpoint,
    paths: tuple[str, ...] | list[str],
    *,
    runner: Runner = subprocess.run,
) -> list[str]:
    if reuse.host is None or destination.host is None or reuse.host != destination.host:
        raise ValueError("hardlink_remote requires two endpoints on the same remote host")
    proc = runner(
        build_remote_python_argv(destination.host),
        input=hardlink_source(reuse.path, destination.path, paths),
        capture_output=True,
        text=True,
        shell=False,
    )
    if int(getattr(proc, "returncode", 1)) != 0:
        detail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip()
        raise RuntimeError(detail or f"remote hard-linking failed on {destination.host}")
    payload = _extract_between(getattr(proc, "stdout", "") or "", LINK_BEGIN, LINK_END, "copy hard-link")
    linked = payload.get("linked")
    if not isinstance(linked, list):
        raise RuntimeError("copy hard-link response did not contain a linked path list")
    return [str(path) for path in linked]


def _extract_between(stdout: str, begin: str, end: str, label: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    begin_indexes = [idx for idx, line in enumerate(lines) if line.strip() == begin]
    if len(begin_indexes) != 1:
        raise RuntimeError(f"{label} sentinel not found uniquely")
    start = begin_indexes[0]
    end_indexes = [idx for idx in range(start + 1, len(lines)) if lines[idx].strip() == end]
    if len(end_indexes) != 1:
        raise RuntimeError(f"{label} sentinel end not found uniquely")
    payload = json.loads("\n".join(lines[start + 1 : end_indexes[0]]))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} payload must be an object")
    return payload


def hardlink_reusable(
    reuse: Endpoint,
    destination: Endpoint,
    paths: tuple[str, ...] | list[str],
    *,
    runner: Runner = subprocess.run,
) -> list[str]:
    if not paths:
        return []
    validate_reuse_endpoints(destination, reuse)
    if destination.host is None:
        return hardlink_local(reuse, destination, paths)
    return hardlink_remote(reuse, destination, paths, runner=runner)
