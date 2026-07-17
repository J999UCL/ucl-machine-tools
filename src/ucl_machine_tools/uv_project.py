"""Local UV project validation, source identity, and managed remote layout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10.
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]

_PYTHON_REQUEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_UV_VERSION_RE = re.compile(r"^uv\s+([A-Za-z0-9][A-Za-z0-9.+-]*)\b")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MANDATORY_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".cache",
        ".uv-cache",
        ".uv_cache",
    }
)
_MANDATORY_TOP_LEVEL_NAMES = frozenset(
    {
        ".ucl-stage-source.json",
        "data",
        "cache",
        "outputs",
        "checkpoints",
        "runs",
        "wandb",
    }
)


class UvProjectError(ValueError):
    """A UV project cannot be staged without violating its local contract."""


@dataclass(frozen=True)
class UvProjectContract:
    """Validated files that define a locked UV project."""

    root: Path
    pyproject_path: Path
    lock_path: Path
    python_version_path: Path
    python_request: str
    lock_sha256: str
    python_version_sha256: str


@dataclass(frozen=True)
class UvTool:
    """The exact local UV executable and release used for staging."""

    executable: Path
    version: str


@dataclass(frozen=True)
class SourceEntry:
    """One portable, content-addressed entry in a project snapshot."""

    path: str
    kind: Literal["file", "symlink", "directory"]
    executable: bool
    size: int
    sha256: str | None = None
    symlink_target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "executable": self.executable,
            "size": self.size,
        }
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.symlink_target is not None:
            payload["symlink_target"] = self.symlink_target
        return payload


@dataclass(frozen=True)
class SourceManifest:
    """Deterministic source snapshot identity independent of traversal metadata."""

    root: Path
    entries: tuple[SourceEntry, ...]
    source_sha256: str
    total_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "file_count": sum(entry.kind == "file" for entry in self.entries),
            "symlink_count": sum(entry.kind == "symlink" for entry in self.entries),
            "directory_count": sum(entry.kind == "directory" for entry in self.entries),
            "total_bytes": self.total_bytes,
            "entries": [entry.as_dict() for entry in self.entries],
        }


@dataclass
class MaterializedSourceSnapshot:
    manifest: SourceManifest
    temporary_root: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.temporary_root, ignore_errors=False)


@dataclass(frozen=True)
class UvProjectSpec:
    """Everything local orchestration needs before performing remote mutation."""

    contract: UvProjectContract
    uv: UvTool
    manifest: SourceManifest

    @property
    def source_sha256(self) -> str:
        return self.manifest.source_sha256

    @property
    def lock_sha256(self) -> str:
        return self.contract.lock_sha256


@dataclass(frozen=True)
class RemoteUvLayout:
    """Safe, content-addressed paths for a managed remote UV stage."""

    remote_root: PurePosixPath
    stage_name: str
    host: str
    uv_version: str
    lock_sha256: str
    source_sha256: str
    setup_environment_sha256: str
    environment_id: str
    stage_id: str
    uv_tools_dir: PurePosixPath
    uv_binary: PurePosixPath
    python_install_dir: PurePosixPath
    uv_cache_dir: PurePosixPath
    sources_dir: PurePosixPath
    source_dir: PurePosixPath
    environments_dir: PurePosixPath
    environment_dir: PurePosixPath
    state_dir: PurePosixPath
    state_file: PurePosixPath
    launchers_dir: PurePosixPath


@dataclass(frozen=True)
class _ScopedIgnore:
    base_parts: tuple[str, ...]
    spec: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise UvProjectError(f"UV project is missing required {label}: {path.name}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise UvProjectError(f"required {label} must be a regular file: {path}")


def _parse_toml(path: Path, label: str) -> None:
    if tomllib is None:
        raise UvProjectError("TOML validation on Python 3.10 requires the 'tomli' package")
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UvProjectError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise UvProjectError(f"invalid {label}: expected a TOML mapping")


def validate_uv_project(project_dir: Path | str) -> UvProjectContract:
    """Validate the three-file local UV contract without changing it."""

    requested_root = Path(project_dir).expanduser()
    try:
        root = requested_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise UvProjectError(f"UV project directory does not exist: {requested_root}") from exc
    if not root.is_dir():
        raise UvProjectError(f"UV project root must be a directory: {root}")

    pyproject = root / "pyproject.toml"
    lock = root / "uv.lock"
    python_version = root / ".python-version"
    for path, label in (
        (pyproject, "pyproject.toml"),
        (lock, "uv.lock"),
        (python_version, ".python-version"),
    ):
        _require_regular_file(path, label)

    _parse_toml(pyproject, "pyproject.toml")
    _parse_toml(lock, "uv.lock")
    try:
        raw_python_request = python_version.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UvProjectError(f"invalid .python-version: {exc}") from exc
    requests = [line.strip() for line in raw_python_request.splitlines() if line.strip()]
    if len(requests) != 1:
        raise UvProjectError(".python-version must contain exactly one Python request")
    python_request = requests[0]
    if not _PYTHON_REQUEST_RE.fullmatch(python_request):
        raise UvProjectError(f"unsafe Python request in .python-version: {python_request!r}")

    return UvProjectContract(
        root=root,
        pyproject_path=pyproject,
        lock_path=lock,
        python_version_path=python_version,
        python_request=python_request,
        lock_sha256=_sha256_file(lock),
        python_version_sha256=_sha256_file(python_version),
    )


def discover_local_uv(
    *,
    runner: Runner = subprocess.run,
    which: Which = shutil.which,
) -> UvTool:
    """Discover and interrogate the exact local UV binary used for staging."""

    discovered = which("uv")
    if not discovered:
        raise UvProjectError("local uv executable was not found on PATH")
    try:
        executable = Path(discovered).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise UvProjectError(f"local uv executable does not exist: {discovered}") from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise UvProjectError(f"local uv path is not executable: {executable}")
    try:
        process = runner(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise UvProjectError(f"could not execute local uv at {executable}: {exc}") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise UvProjectError(detail or f"local uv --version exited {process.returncode}")
    output = (process.stdout or "").strip()
    match = _UV_VERSION_RE.match(output)
    if match is None:
        raise UvProjectError(f"could not parse local uv version from: {output!r}")
    return UvTool(executable=executable, version=match.group(1))


def check_uv_lock(
    contract: UvProjectContract,
    uv: UvTool,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Require the committed lock to be current without modifying it."""

    argv = [
        str(uv.executable),
        "lock",
        "--check",
        "--project",
        str(contract.root),
    ]
    try:
        process = runner(argv, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise UvProjectError(f"failed to check UV lockfile: {exc}") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise UvProjectError(detail or f"uv lock --check exited {process.returncode}")


def _pathspec_type() -> type[Any]:
    try:
        from pathspec import GitIgnoreSpec
    except ModuleNotFoundError as exc:
        raise UvProjectError(
            "source manifest generation requires the 'pathspec' package"
        ) from exc
    return GitIgnoreSpec


def _load_ignore(path: Path, base_parts: tuple[str, ...]) -> _ScopedIgnore | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise UvProjectError(f"could not read ignore file {path}: {exc}") from exc
    return _ScopedIgnore(base_parts=base_parts, spec=_pathspec_type().from_lines(lines))


def _scoped_ignore_decision(
    scopes: Iterable[_ScopedIgnore],
    relative_parts: tuple[str, ...],
    *,
    is_directory: bool,
) -> bool:
    ignored: bool | None = None
    for scope in scopes:
        base_length = len(scope.base_parts)
        if relative_parts[:base_length] != scope.base_parts:
            continue
        local_parts = relative_parts[base_length:]
        if not local_parts:
            continue
        local_path = "/".join(local_parts) + ("/" if is_directory else "")
        result = scope.spec.check_file(local_path)
        if result.include is not None:
            ignored = bool(result.include)
    return ignored is True


def _is_secret_env_name(name: str) -> bool:
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def _is_mandatory_exclusion(relative_parts: tuple[str, ...]) -> bool:
    if not relative_parts:
        return False
    if relative_parts[0] in _MANDATORY_TOP_LEVEL_NAMES:
        return True
    for part in relative_parts:
        if part in _MANDATORY_DIRECTORY_NAMES or _is_secret_env_name(part):
            return True
        if part.endswith((".pyc", ".pyo")):
            return True
    return False


def _safe_symlink_target(root: Path, link: Path, target: str) -> None:
    target_path = Path(target)
    if target_path.is_absolute():
        raise UvProjectError(f"absolute symlink is not portable: {link.relative_to(root)} -> {target}")
    try:
        resolved = (link.parent / target_path).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise UvProjectError(f"invalid symlink target: {link.relative_to(root)} -> {target}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UvProjectError(
            f"symlink escapes project root: {link.relative_to(root)} -> {target}"
        ) from exc


def _file_entry(root: Path, path: Path, info: os.stat_result) -> SourceEntry:
    return SourceEntry(
        path=path.relative_to(root).as_posix(),
        kind="file",
        executable=bool(stat.S_IMODE(info.st_mode) & 0o111),
        size=info.st_size,
        sha256=_sha256_file(path),
    )


def _symlink_entry(root: Path, path: Path) -> SourceEntry:
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise UvProjectError(f"could not read symlink {path.relative_to(root)}: {exc}") from exc
    _safe_symlink_target(root, path, target)
    return SourceEntry(
        path=path.relative_to(root).as_posix(),
        kind="symlink",
        executable=False,
        size=len(os.fsencode(target)),
        symlink_target=target,
    )


def _directory_entry(root: Path, path: Path, info: os.stat_result) -> SourceEntry:
    executable = bool(stat.S_IMODE(info.st_mode) & 0o111)
    if not executable:
        raise UvProjectError(f"source directory is not traversable: {path.relative_to(root)}")
    return SourceEntry(
        path=path.relative_to(root).as_posix(),
        kind="directory",
        executable=executable,
        size=0,
    )


def _source_identity(entries: Iterable[SourceEntry]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ucl-source-manifest-v1\n")
    for entry in entries:
        encoded = json.dumps(
            entry.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def build_source_manifest(project_dir: Path | str) -> SourceManifest:
    """Build a deterministic, ignore-aware identity for a project working tree."""

    requested_root = Path(project_dir).expanduser()
    try:
        root = requested_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise UvProjectError(f"source directory does not exist: {requested_root}") from exc
    if not root.is_dir():
        raise UvProjectError(f"source root must be a directory: {root}")

    entries: list[SourceEntry] = []

    def visit(
        directory: Path,
        relative_parts: tuple[str, ...],
        inherited_git: tuple[_ScopedIgnore, ...],
        inherited_ucl: tuple[_ScopedIgnore, ...],
    ) -> None:
        git_scopes = list(inherited_git)
        ucl_scopes = list(inherited_ucl)
        git_ignore = _load_ignore(directory / ".gitignore", relative_parts)
        ucl_ignore = _load_ignore(directory / ".uclignore", relative_parts)
        if git_ignore is not None:
            git_scopes.append(git_ignore)
        if ucl_ignore is not None:
            ucl_scopes.append(ucl_ignore)

        try:
            with os.scandir(directory) as scan:
                children = sorted(scan, key=lambda item: item.name)
        except OSError as exc:
            relative = "/".join(relative_parts) or "."
            raise UvProjectError(f"could not scan source directory {relative}: {exc}") from exc
        for child in children:
            child_parts = (*relative_parts, child.name)
            if _is_mandatory_exclusion(child_parts):
                continue
            path = Path(child.path)
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise UvProjectError(f"could not inspect source entry {'/'.join(child_parts)}: {exc}") from exc
            is_directory = stat.S_ISDIR(info.st_mode)
            if _scoped_ignore_decision(git_scopes, child_parts, is_directory=is_directory):
                continue
            if _scoped_ignore_decision(ucl_scopes, child_parts, is_directory=is_directory):
                continue

            if stat.S_ISLNK(info.st_mode):
                entries.append(_symlink_entry(root, path))
            elif is_directory:
                entries.append(_directory_entry(root, path, info))
                visit(path, child_parts, tuple(git_scopes), tuple(ucl_scopes))
            elif stat.S_ISREG(info.st_mode):
                entries.append(_file_entry(root, path, info))
            else:
                raise UvProjectError(
                    f"unsupported special file in source tree: {'/'.join(child_parts)}"
                )

    visit(root, (), (), ())
    ordered = tuple(sorted(entries, key=lambda entry: entry.path))
    manifest = SourceManifest(
        root=root,
        entries=ordered,
        source_sha256=_source_identity(ordered),
        total_bytes=sum(entry.size for entry in ordered if entry.kind == "file"),
    )
    entry_by_path = {entry.path: entry for entry in manifest.entries}
    for required in ("pyproject.toml", "uv.lock", ".python-version"):
        entry = entry_by_path.get(required)
        if entry is None or entry.kind != "file":
            raise UvProjectError(f"required UV contract file is excluded from the source snapshot: {required}")
    return manifest


def materialize_source_snapshot(manifest: SourceManifest) -> MaterializedSourceSnapshot:
    """Copy the verified manifest bytes into a stable local transfer tree."""

    temporary_root = Path(tempfile.mkdtemp(prefix="ucl-stage-source-"))
    try:
        for entry in manifest.entries:
            source = manifest.root / entry.path
            destination = temporary_root / entry.path
            if entry.kind == "directory":
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o755 if entry.executable else 0o644)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "symlink":
                current_target = os.readlink(source)
                if current_target != entry.symlink_target:
                    raise UvProjectError(f"source changed while staging: {entry.path}")
                destination.symlink_to(current_target)
                continue
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(source, flags)
            except OSError as exc:
                raise UvProjectError(f"source changed while staging: {entry.path}") from exc
            with os.fdopen(descriptor, "rb") as source_handle, destination.open("xb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            destination.chmod(0o755 if entry.executable else 0o644)
            if destination.stat().st_size != entry.size or _sha256_file(destination) != entry.sha256:
                raise UvProjectError(f"source changed while staging: {entry.path}")
        stable = SourceManifest(
            root=temporary_root,
            entries=manifest.entries,
            source_sha256=manifest.source_sha256,
            total_bytes=manifest.total_bytes,
        )
        return MaterializedSourceSnapshot(manifest=stable, temporary_root=temporary_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def load_uv_project(
    project_dir: Path | str,
    *,
    runner: Runner = subprocess.run,
    which: Which = shutil.which,
) -> UvProjectSpec:
    """Resolve all local stage inputs, checking the lock before snapshot work."""

    contract = validate_uv_project(project_dir)
    uv = discover_local_uv(runner=runner, which=which)
    check_uv_lock(contract, uv, runner=runner)
    manifest = build_source_manifest(contract.root)
    lock_entry = next(entry for entry in manifest.entries if entry.path == "uv.lock")
    if lock_entry.sha256 != contract.lock_sha256:
        raise UvProjectError("uv.lock changed while staging; retry from a stable working tree")
    python_entry = next(entry for entry in manifest.entries if entry.path == ".python-version")
    if python_entry.sha256 != contract.python_version_sha256:
        raise UvProjectError(".python-version changed while staging; retry from a stable working tree")
    return UvProjectSpec(contract=contract, uv=uv, manifest=manifest)


def _validate_remote_root(value: str | PurePosixPath) -> PurePosixPath:
    raw = str(value)
    if not raw.startswith("/"):
        raise UvProjectError(f"remote_root must be absolute: {raw!r}")
    path = PurePosixPath(raw)
    if path == PurePosixPath("/"):
        raise UvProjectError("remote_root must not be filesystem root")
    if ".." in path.parts:
        raise UvProjectError("remote_root must not contain '..'")
    if any(character in raw for character in ("\x00", "\n", "\r")):
        raise UvProjectError("remote_root contains unsafe control characters")
    return path


def _validate_token(value: str, label: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise UvProjectError(f"unsafe {label}: {value!r}")


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise UvProjectError(f"{label} must be a lowercase SHA-256 digest")


def hash_setup_environment(environment: Iterable[tuple[str, str]]) -> str:
    """Hash setup-affecting environment values without persisting them."""

    normalized: dict[str, str] = {}
    for key, value in environment:
        if key in normalized:
            raise UvProjectError(f"duplicate setup environment key: {key}")
        normalized[key] = value
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_remote_layout(
    *,
    remote_root: str | PurePosixPath,
    stage_name: str,
    host: str,
    uv_version: str,
    lock_sha256: str,
    source_sha256: str,
    setup_environment_sha256: str,
) -> RemoteUvLayout:
    """Derive deterministic managed paths without consulting remote state."""

    root = _validate_remote_root(remote_root)
    _validate_token(stage_name, "stage name", _SAFE_NAME_RE)
    _validate_token(host, "host", _SAFE_HOST_RE)
    _validate_token(uv_version, "uv version", _SAFE_VERSION_RE)
    _validate_sha256(lock_sha256, "lock_sha256")
    _validate_sha256(source_sha256, "source_sha256")
    _validate_sha256(setup_environment_sha256, "setup_environment_sha256")

    identity_payload = json.dumps(
        {
            "uv_version": uv_version,
            "lock_sha256": lock_sha256,
            "source_sha256": source_sha256,
            "setup_environment_sha256": setup_environment_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    environment_id = hashlib.sha256(identity_payload).hexdigest()
    stage_identity = hashlib.sha256(
        json.dumps(
            {"environment_id": environment_id, "remote_root": str(root)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    stage_id = f"{stage_name}-{host}-{stage_identity[:16]}"
    stage_root = root / "stages" / stage_name
    uv_tools_dir = root / "tools" / "uv" / uv_version
    sources_dir = stage_root / "sources"
    environments_dir = stage_root / "envs"
    state_dir = stage_root / "state"

    return RemoteUvLayout(
        remote_root=root,
        stage_name=stage_name,
        host=host,
        uv_version=uv_version,
        lock_sha256=lock_sha256,
        source_sha256=source_sha256,
        setup_environment_sha256=setup_environment_sha256,
        environment_id=environment_id,
        stage_id=stage_id,
        uv_tools_dir=uv_tools_dir,
        uv_binary=uv_tools_dir / "uv",
        python_install_dir=root / "tools" / "python",
        uv_cache_dir=root / "cache" / "uv",
        sources_dir=sources_dir,
        source_dir=sources_dir / source_sha256,
        environments_dir=environments_dir,
        environment_dir=environments_dir / environment_id,
        state_dir=state_dir,
        state_file=state_dir / f"{stage_id}.json",
        launchers_dir=root / "launchers",
    )
