"""Pure primitives for preparing and using locked uv projects remotely."""

from __future__ import annotations

import base64
import json
import posixpath
import re
import shlex
from dataclasses import dataclass


SCHEMA_VERSION = 1
SETUP_SENTINEL_BEGIN = "UCL_UV_SETUP_JSON_BEGIN"
SETUP_SENTINEL_END = "UCL_UV_SETUP_JSON_END"
TSG_PYTHON_SETUP = "/opt/Python/Python-3.11.5_Setup.csh"
_UV_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?(?:[-+][A-Za-z0-9.-]+)?$")
_GPU_ID_RE = re.compile(r"^[0-9]+$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_REQUEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_RESERVED_SETUP_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON_INSTALL_DIR",
}


def _validate_absolute_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute POSIX path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{label} must not contain control characters")
    if not value.startswith("/"):
        raise ValueError(f"{label} must be absolute: {value!r}")
    components = value.split("/")
    if ".." in components:
        raise ValueError(f"{label} must not contain '..': {value!r}")
    normalized = posixpath.normpath(value)
    if normalized == "/":
        raise ValueError(f"{label} must not be the filesystem root")
    return normalized


def _is_within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip("/") + "/")


@dataclass(frozen=True)
class UvRemotePaths:
    """All persistent paths managed by one remote uv setup."""

    source_dir: str
    environment_dir: str
    uv_cache_dir: str
    uv_tool_dir: str
    uv_binary_path: str
    python_install_dir: str
    ready_state_path: str
    failed_state_path: str
    log_path: str
    environment_lock_path: str
    uv_tool_lock_path: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            normalized = _validate_absolute_path(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)

        if not _is_within(self.uv_binary_path, self.uv_tool_dir) or self.uv_binary_path == self.uv_tool_dir:
            raise ValueError("uv_binary_path must be a file beneath uv_tool_dir")

        protected_from_environment_cleanup = {
            "source_dir": self.source_dir,
            "uv_cache_dir": self.uv_cache_dir,
            "uv_tool_dir": self.uv_tool_dir,
            "uv_binary_path": self.uv_binary_path,
            "python_install_dir": self.python_install_dir,
            "ready_state_path": self.ready_state_path,
            "failed_state_path": self.failed_state_path,
            "log_path": self.log_path,
            "environment_lock_path": self.environment_lock_path,
            "uv_tool_lock_path": self.uv_tool_lock_path,
        }
        for label, value in protected_from_environment_cleanup.items():
            if _is_within(value, self.environment_dir):
                raise ValueError(f"{label} must not be inside environment_dir")

        files = {
            self.uv_binary_path,
            self.ready_state_path,
            self.failed_state_path,
            self.log_path,
            self.environment_lock_path,
            self.uv_tool_lock_path,
        }
        if len(files) != 6:
            raise ValueError("managed remote file paths must be distinct")


@dataclass(frozen=True)
class UvSetupSpec:
    """Validated inputs for an exact remote uv environment setup."""

    uv_version: str
    paths: UvRemotePaths
    source_sha256: str
    lock_sha256: str
    python_request: str
    gpu_id: str | None = None
    cuda_visibility_script: str | None = None
    setup_env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.uv_version, str) or not _UV_VERSION_RE.fullmatch(self.uv_version):
            raise ValueError(f"uv_version must be an exact numeric release: {self.uv_version!r}")
        if not isinstance(self.paths, UvRemotePaths):
            raise ValueError("paths must be UvRemotePaths")
        for value, label in (
            (self.source_sha256, "source_sha256"),
            (self.lock_sha256, "lock_sha256"),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if not isinstance(self.python_request, str) or _PYTHON_REQUEST_RE.fullmatch(self.python_request) is None:
            raise ValueError(f"invalid python_request: {self.python_request!r}")
        if self.gpu_id is not None and (
            not isinstance(self.gpu_id, str) or not _GPU_ID_RE.fullmatch(self.gpu_id)
        ):
            raise ValueError(f"gpu_id must be a selected numeric GPU index: {self.gpu_id!r}")
        if self.gpu_id is None:
            if self.cuda_visibility_script is not None:
                raise ValueError("cuda_visibility_script requires gpu_id")
        else:
            if self.cuda_visibility_script is None:
                raise ValueError("cuda_visibility_script is required when gpu_id is selected")
            normalized = _validate_absolute_path(self.cuda_visibility_script, "cuda_visibility_script")
            object.__setattr__(self, "cuda_visibility_script", normalized)
        normalized_env: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in self.setup_env:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("setup_env entries must be (KEY, VALUE) pairs")
            key, value = item
            if not isinstance(key, str) or _ENV_KEY_RE.fullmatch(key) is None:
                raise ValueError(f"invalid setup environment key: {key!r}")
            if key in seen:
                raise ValueError(f"duplicate setup environment key: {key}")
            if key.startswith("UCL_") or key in _RESERVED_SETUP_ENV:
                raise ValueError(f"setup environment key is managed by ucl: {key}")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"setup environment value for {key!r} must be a string without NUL bytes")
            seen.add(key)
            normalized_env.append((key, value))
        object.__setattr__(self, "setup_env", tuple(normalized_env))


@dataclass(frozen=True)
class UvSetupPayload:
    """Generated files and argv suitable for an asynchronous tmux launcher."""

    spec: UvSetupSpec
    csh_driver_path: str
    bash_driver_path: str
    csh_source: str
    bash_source: str

    @property
    def entrypoint(self) -> tuple[str, ...]:
        return ("csh", "-f", self.csh_driver_path)

    @property
    def files(self) -> dict[str, str]:
        return {
            self.csh_driver_path: self.csh_source,
            self.bash_driver_path: self.bash_source,
        }


@dataclass(frozen=True)
class UvSetupResult:
    """Validated setup state emitted by the remote driver."""

    schema_version: int
    ok: bool
    status: str
    phase: str
    uv_version: str
    source_sha256: str
    lock_sha256: str
    python_request: str
    python_path: str
    source_dir: str
    environment_dir: str
    uv_binary_path: str
    ready_state_path: str
    failed_state_path: str
    log_path: str
    reused_uv: bool
    reused_environment: bool
    returncode: int
    error: str
    failed_command: str
    failed_line: int | None


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _csh_decode_assignment(name: str, value: str) -> str:
    encoded = _b64(value)
    decoder = f'import base64;print(base64.b64decode("{encoded}").decode("utf-8"))'
    return f"set {name} = `python3 -c {shlex.quote(decoder)}`"


def _bash_assignment(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _relative_binary_path(paths: UvRemotePaths) -> str:
    relative = posixpath.relpath(paths.uv_binary_path, paths.uv_tool_dir)
    if relative == ".." or relative.startswith("../"):
        raise ValueError("uv_binary_path must be beneath uv_tool_dir")
    return relative


def _build_csh_source(spec: UvSetupSpec, bash_driver_path: str) -> str:
    lines = [
        "#!/bin/csh -f",
        "source /opt/Python/Python-3.11.5_Setup.csh",
        "if ($status != 0) setenv UCL_TSG_BOOTSTRAP_ERROR python_setup_failed",
    ]
    if spec.gpu_id is not None:
        assert spec.cuda_visibility_script is not None
        lines.extend(
            [
                _csh_decode_assignment("ucl_cuda_setup", spec.cuda_visibility_script),
                'source "$ucl_cuda_setup"',
                "if ($status != 0) setenv UCL_TSG_BOOTSTRAP_ERROR cuda_visibility_setup_failed",
                f"setenv CUDA_VISIBLE_DEVICES {spec.gpu_id}",
            ]
        )
    lines.extend(
        [
            _csh_decode_assignment("ucl_bash_driver", bash_driver_path),
            'exec /bin/bash --noprofile --norc "$ucl_bash_driver"',
            "",
        ]
    )
    return "\n".join(lines)


def _build_bash_source(spec: UvSetupSpec) -> str:
    paths = spec.paths
    assignments = "\n".join(
        [
            _bash_assignment("UCL_UV_VERSION", spec.uv_version),
            _bash_assignment("UCL_SOURCE_SHA256", spec.source_sha256),
            _bash_assignment("UCL_LOCK_SHA256", spec.lock_sha256),
            _bash_assignment("UCL_PYTHON_REQUEST", spec.python_request),
            _bash_assignment("UCL_SOURCE_DIR", paths.source_dir),
            _bash_assignment("UCL_ENVIRONMENT_DIR", paths.environment_dir),
            _bash_assignment("UCL_UV_CACHE_DIR", paths.uv_cache_dir),
            _bash_assignment("UCL_UV_TOOL_DIR", paths.uv_tool_dir),
            _bash_assignment("UCL_UV_BINARY", paths.uv_binary_path),
            _bash_assignment("UCL_UV_BINARY_RELATIVE", _relative_binary_path(paths)),
            _bash_assignment("UCL_PYTHON_INSTALL_DIR", paths.python_install_dir),
            _bash_assignment("UCL_READY_STATE", paths.ready_state_path),
            _bash_assignment("UCL_FAILED_STATE", paths.failed_state_path),
            _bash_assignment("UCL_LOG_PATH", paths.log_path),
            _bash_assignment("UCL_ENVIRONMENT_LOCK", paths.environment_lock_path),
            _bash_assignment("UCL_UV_TOOL_LOCK", paths.uv_tool_lock_path),
            _bash_assignment(
                "UCL_UV_INSTALLER_URL",
                f"https://astral.sh/uv/{spec.uv_version}/install.sh",
            ),
            _bash_assignment("UCL_SENTINEL_BEGIN", SETUP_SENTINEL_BEGIN),
            _bash_assignment("UCL_SENTINEL_END", SETUP_SENTINEL_END),
        ]
    )
    setup_exports = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in spec.setup_env
    )
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail

{assignments}
{setup_exports}
CURRENT_PHASE=initializing
REUSED_UV=false
REUSED_ENVIRONMENT=false
UCL_ENVIRONMENT_CREATED=false
UCL_PYTHON_PATH=""
ucl_tool_temp=""
ucl_installer=""

mkdir -p -- "$(dirname -- "$UCL_LOG_PATH")"
exec > >(tee -a "$UCL_LOG_PATH") 2>&1

ucl_write_state() {{
  local target="$1"
  local ok="$2"
  local status="$3"
  local returncode="$4"
  local error="$5"
  local failed_command="$6"
  local failed_line="$7"
  python3 - "$target" "$ok" "$status" "$CURRENT_PHASE" "$returncode" "$error" "$failed_command" "$failed_line" \
    "$REUSED_UV" "$REUSED_ENVIRONMENT" "$UCL_UV_VERSION" "$UCL_SOURCE_SHA256" "$UCL_LOCK_SHA256" \
    "$UCL_PYTHON_REQUEST" "$UCL_PYTHON_PATH" "$UCL_SOURCE_DIR" "$UCL_ENVIRONMENT_DIR" \
    "$UCL_UV_BINARY" "$UCL_READY_STATE" "$UCL_FAILED_STATE" "$UCL_LOG_PATH" \
    "$UCL_SENTINEL_BEGIN" "$UCL_SENTINEL_END" <<'PY'
import datetime
import json
import os
import pathlib
import sys

(
    target,
    ok,
    status,
    phase,
    returncode,
    error,
    failed_command,
    failed_line,
    reused_uv,
    reused_environment,
    uv_version,
    source_sha256,
    lock_sha256,
    python_request,
    python_path,
    source_dir,
    environment_dir,
    uv_binary_path,
    ready_state_path,
    failed_state_path,
    log_path,
    sentinel_begin,
    sentinel_end,
) = sys.argv[1:]
payload = {{
    "schema_version": 1,
    "ok": ok == "true",
    "status": status,
    "phase": phase,
    "uv_version": uv_version,
    "source_sha256": source_sha256,
    "lock_sha256": lock_sha256,
    "python_request": python_request,
    "python_path": python_path,
    "source_dir": source_dir,
    "environment_dir": environment_dir,
    "uv_binary_path": uv_binary_path,
    "ready_state_path": ready_state_path,
    "failed_state_path": failed_state_path,
    "log_path": log_path,
    "reused_uv": reused_uv == "true",
    "reused_environment": reused_environment == "true",
    "returncode": int(returncode),
    "error": error,
    "failed_command": failed_command,
    "failed_line": int(failed_line) if failed_line else None,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}}
destination = pathlib.Path(target)
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + f".tmp.{{os.getpid()}}")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
print(sentinel_begin)
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
print(sentinel_end)
PY
}}

ucl_cleanup_temporary() {{
  if [[ -n "$ucl_installer" ]]; then
    rm -f -- "$ucl_installer"
  fi
  if [[ -n "$ucl_tool_temp" ]]; then
    rm -rf -- "$ucl_tool_temp"
  fi
  if [[ "$UCL_ENVIRONMENT_CREATED" == true ]]; then
    rm -rf -- "$UCL_ENVIRONMENT_DIR"
  fi
}}

ucl_fail() {{
  local returncode="$1"
  local failed_line="$2"
  local failed_command="$3"
  trap - ERR
  set +e
  ucl_cleanup_temporary
  rm -f -- "$UCL_READY_STATE"
  local error="phase $CURRENT_PHASE failed at line $failed_line: $failed_command (exit $returncode)"
  ucl_write_state "$UCL_FAILED_STATE" false failed "$returncode" "$error" "$failed_command" "$failed_line"
  exit "$returncode"
}}

ucl_abort() {{
  local returncode="$1"
  local error="$2"
  trap - ERR
  set +e
  ucl_cleanup_temporary
  rm -f -- "$UCL_READY_STATE"
  ucl_write_state "$UCL_FAILED_STATE" false failed "$returncode" "$error" "" ""
  exit "$returncode"
}}

ucl_cancel() {{
  local returncode="$1"
  local signal_name="$2"
  trap - ERR INT TERM
  set +e
  ucl_cleanup_temporary
  rm -f -- "$UCL_READY_STATE"
  ucl_write_state "$UCL_FAILED_STATE" false failed "$returncode" "setup cancelled by $signal_name" "" ""
  exit "$returncode"
}}

trap 'ucl_fail $? $LINENO "$BASH_COMMAND"' ERR
trap 'ucl_cancel 130 INT' INT
trap 'ucl_cancel 143 TERM' TERM

if [[ -n "${{UCL_TSG_BOOTSTRAP_ERROR:-}}" ]]; then
  ucl_abort 69 "TSG bootstrap failed: $UCL_TSG_BOOTSTRAP_ERROR"
fi

CURRENT_PHASE=preflight
for ucl_required in python3 curl flock mktemp; do
  command -v "$ucl_required" >/dev/null 2>&1 || ucl_abort 70 "required command not found: $ucl_required"
done
[[ -d "$UCL_SOURCE_DIR" ]] || ucl_abort 66 "source directory does not exist: $UCL_SOURCE_DIR"
for ucl_required_file in pyproject.toml uv.lock .python-version; do
  [[ -f "$UCL_SOURCE_DIR/$ucl_required_file" ]] || ucl_abort 66 "required project file does not exist: $UCL_SOURCE_DIR/$ucl_required_file"
done

mkdir -p -- "$UCL_UV_CACHE_DIR" "$UCL_PYTHON_INSTALL_DIR" \
  "$(dirname -- "$UCL_UV_TOOL_DIR")" "$(dirname -- "$UCL_ENVIRONMENT_DIR")" \
  "$(dirname -- "$UCL_READY_STATE")" "$(dirname -- "$UCL_FAILED_STATE")" \
  "$(dirname -- "$UCL_ENVIRONMENT_LOCK")" "$(dirname -- "$UCL_UV_TOOL_LOCK")"

export UV_CACHE_DIR="$UCL_UV_CACHE_DIR"
export UV_PROJECT_ENVIRONMENT="$UCL_ENVIRONMENT_DIR"
export UV_PYTHON_INSTALL_DIR="$UCL_PYTHON_INSTALL_DIR"

ucl_uv_version_is_exact() {{
  local binary="$1"
  [[ -x "$binary" ]] && [[ "$("$binary" --version)" == "uv $UCL_UV_VERSION" ]]
}}

CURRENT_PHASE=uv_lock_wait
exec 9>"$UCL_UV_TOOL_LOCK"
flock -w 1800 9 || ucl_abort 75 "timed out waiting for uv tool lock"

CURRENT_PHASE=uv_bootstrap
if ucl_uv_version_is_exact "$UCL_UV_BINARY"; then
  REUSED_UV=true
else
  if [[ -e "$UCL_UV_TOOL_DIR" ]]; then
    ucl_abort 65 "existing uv tool directory does not contain uv $UCL_UV_VERSION: $UCL_UV_TOOL_DIR"
  fi
  ucl_tool_temp="$(mktemp -d "${{UCL_UV_TOOL_DIR}}.tmp.XXXXXX")"
  ucl_installer="$(mktemp "${{UCL_UV_TOOL_DIR}}.installer.XXXXXX")"
  curl --fail --location --silent --show-error "$UCL_UV_INSTALLER_URL" -o "$ucl_installer"
  UV_UNMANAGED_INSTALL="$ucl_tool_temp" /bin/sh "$ucl_installer"
  ucl_uv_candidate="$ucl_tool_temp/$UCL_UV_BINARY_RELATIVE"
  CURRENT_PHASE=uv_verify
  if ! ucl_uv_version_is_exact "$ucl_uv_candidate"; then
    ucl_abort 65 "installed uv binary did not report exact version uv $UCL_UV_VERSION"
  fi
  rm -f -- "$ucl_installer"
  ucl_installer=""
  mv -- "$ucl_tool_temp" "$UCL_UV_TOOL_DIR"
  ucl_tool_temp=""
fi
exec 9>&-

CURRENT_PHASE=environment_lock_wait
exec 8>"$UCL_ENVIRONMENT_LOCK"
flock -w 1800 8 || ucl_abort 75 "timed out waiting for environment lock"
cd -- "$UCL_SOURCE_DIR"

ucl_ready_matches() {{
  python3 - "$UCL_READY_STATE" "$UCL_UV_VERSION" "$UCL_SOURCE_SHA256" "$UCL_LOCK_SHA256" \
    "$UCL_PYTHON_REQUEST" "$UCL_SOURCE_DIR" "$UCL_ENVIRONMENT_DIR" "$UCL_UV_BINARY" <<'PY'
import json
import pathlib
import sys

path, uv_version, source_sha256, lock_sha256, python_request, source_dir, environment_dir, uv_binary_path = sys.argv[1:]
try:
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {{
    "schema_version": 1,
    "ok": True,
    "status": "ready",
    "phase": "ready",
    "uv_version": uv_version,
    "source_sha256": source_sha256,
    "lock_sha256": lock_sha256,
    "python_request": python_request,
    "source_dir": source_dir,
    "environment_dir": environment_dir,
    "uv_binary_path": uv_binary_path,
}}
raise SystemExit(0 if all(payload.get(key) == value for key, value in expected.items()) else 1)
PY
}}

if [[ -e "$UCL_READY_STATE" ]]; then
  CURRENT_PHASE=reuse_verify
  ucl_ready_matches || ucl_abort 65 "ready state does not match requested uv setup: $UCL_READY_STATE"
  [[ -d "$UCL_ENVIRONMENT_DIR" ]] || ucl_abort 65 "ready environment directory is missing: $UCL_ENVIRONMENT_DIR"
  "$UCL_UV_BINARY" lock --check
  "$UCL_UV_BINARY" sync --frozen --check
  REUSED_ENVIRONMENT=true
else
  CURRENT_PHASE=lock_check
  "$UCL_UV_BINARY" lock --check
  if [[ -L "$UCL_ENVIRONMENT_DIR" ]]; then
    ucl_abort 65 "environment path must not be a symlink: $UCL_ENVIRONMENT_DIR"
  fi
  if [[ -e "$UCL_ENVIRONMENT_DIR" ]]; then
    rm -rf -- "$UCL_ENVIRONMENT_DIR"
  fi
  UCL_ENVIRONMENT_CREATED=true
  CURRENT_PHASE=sync
  "$UCL_UV_BINARY" sync --frozen --no-editable
  CURRENT_PHASE=sync_check
  "$UCL_UV_BINARY" sync --frozen --check
  UCL_ENVIRONMENT_CREATED=false
fi

CURRENT_PHASE=interpreter_verify
UCL_PYTHON_PATH="$UCL_ENVIRONMENT_DIR/bin/python"
[[ -x "$UCL_PYTHON_PATH" ]] || ucl_abort 65 "environment interpreter is missing: $UCL_PYTHON_PATH"
"$UCL_PYTHON_PATH" -c 'import sys; print(sys.executable)'

CURRENT_PHASE=ready
rm -f -- "$UCL_FAILED_STATE"
ucl_write_state "$UCL_READY_STATE" true ready 0 "" "" ""
'''


def build_setup_payload(
    spec: UvSetupSpec,
    *,
    csh_driver_path: str,
    bash_driver_path: str,
) -> UvSetupPayload:
    """Generate setup files without performing remote work."""

    if not isinstance(spec, UvSetupSpec):
        raise ValueError("spec must be UvSetupSpec")
    csh_path = _validate_absolute_path(csh_driver_path, "csh_driver_path")
    bash_path = _validate_absolute_path(bash_driver_path, "bash_driver_path")
    if csh_path == bash_path:
        raise ValueError("csh_driver_path and bash_driver_path must be distinct")
    return UvSetupPayload(
        spec=spec,
        csh_driver_path=csh_path,
        bash_driver_path=bash_path,
        csh_source=_build_csh_source(spec, bash_path),
        bash_source=_build_bash_source(spec),
    )


_RESULT_STRING_FIELDS = (
    "status",
    "phase",
    "uv_version",
    "source_sha256",
    "lock_sha256",
    "python_request",
    "python_path",
    "source_dir",
    "environment_dir",
    "uv_binary_path",
    "ready_state_path",
    "failed_state_path",
    "log_path",
    "error",
    "failed_command",
)


def _validate_state_payload(payload: object) -> UvSetupResult:
    if not isinstance(payload, dict):
        raise ValueError("uv setup state must be a JSON object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported uv setup state schema: {payload.get('schema_version')!r}")
    if type(payload.get("ok")) is not bool:
        raise ValueError("uv setup state field 'ok' must be boolean")
    for field in _RESULT_STRING_FIELDS:
        if not isinstance(payload.get(field), str):
            raise ValueError(f"uv setup state field {field!r} must be a string")
    for field in ("source_sha256", "lock_sha256"):
        if _SHA256_RE.fullmatch(payload[field]) is None:
            raise ValueError(f"uv setup state field {field!r} must be SHA-256")
    if _PYTHON_REQUEST_RE.fullmatch(payload["python_request"]) is None:
        raise ValueError("uv setup state field 'python_request' is invalid")
    if payload["status"] not in {"ready", "failed"}:
        raise ValueError(f"uv setup state has invalid status: {payload['status']!r}")
    for field in ("reused_uv", "reused_environment"):
        if type(payload.get(field)) is not bool:
            raise ValueError(f"uv setup state field {field!r} must be boolean")
    if type(payload.get("returncode")) is not int:
        raise ValueError("uv setup state field 'returncode' must be an integer")
    failed_line = payload.get("failed_line")
    if failed_line is not None and type(failed_line) is not int:
        raise ValueError("uv setup state field 'failed_line' must be an integer or null")
    if payload["ok"] != (payload["status"] == "ready"):
        raise ValueError("uv setup state ok/status fields contradict each other")
    if payload["ok"]:
        if payload["returncode"] != 0:
            raise ValueError("ready uv setup state must have returncode 0")
        if payload["phase"] != "ready":
            raise ValueError("ready uv setup state must have phase 'ready'")
        if payload["error"] or payload["failed_command"] or failed_line is not None:
            raise ValueError("ready uv setup state must not contain failure details")
        if not payload["python_path"].startswith("/"):
            raise ValueError("ready uv setup state must include an absolute python_path")
    else:
        if payload["returncode"] == 0:
            raise ValueError("failed uv setup state must have nonzero returncode")
        if not payload["error"]:
            raise ValueError("failed uv setup state must include error details")
    return UvSetupResult(
        **{field: payload[field] for field in UvSetupResult.__dataclass_fields__}
    )


def parse_state_json(text: str) -> UvSetupResult:
    """Parse and validate a ready or failed state file."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("uv setup state is not valid JSON") from exc
    return _validate_state_payload(payload)


def parse_setup_result(stdout: str) -> UvSetupResult:
    """Parse one sentinel result while ignoring all output outside it."""

    lines = stdout.splitlines()
    begin_indices = [index for index, line in enumerate(lines) if line.strip() == SETUP_SENTINEL_BEGIN]
    if not begin_indices:
        raise ValueError("uv setup sentinel not found")
    if len(begin_indices) != 1:
        raise ValueError("uv setup sentinel must appear exactly once")
    begin = begin_indices[0]
    end_indices = [index for index, line in enumerate(lines) if line.strip() == SETUP_SENTINEL_END]
    if not end_indices:
        raise ValueError("uv setup sentinel end not found")
    if len(end_indices) != 1:
        raise ValueError("uv setup sentinel end must appear exactly once")
    end = end_indices[0]
    if end <= begin:
        raise ValueError("uv setup sentinel end appears before its begin marker")
    payload_lines = lines[begin + 1 : end]
    if len(payload_lines) != 1:
        raise ValueError("uv setup sentinel payload must be exactly one JSON line")
    return parse_state_json(payload_lines[0])
