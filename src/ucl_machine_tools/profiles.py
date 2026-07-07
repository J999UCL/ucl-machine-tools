"""Launch profile loading and remote script generation."""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_FIELDS = {
    "description",
    "env",
    "extends",
    "preflight",
    "preflight_after_setup",
    "run_prefix",
    "shell",
    "source",
}
ALLOWED_SHELLS = {"bash", "csh-bootstrap"}


@dataclass(frozen=True)
class CheckCommand:
    label: str
    command: str


@dataclass(frozen=True)
class ProfileDef:
    name: str
    description: str = ""
    extends: tuple[str, ...] = ()
    shell: str | None = None
    source: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    preflight: tuple[CheckCommand, ...] = ()
    preflight_after_setup: tuple[CheckCommand, ...] = ()
    run_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedProfile:
    names: tuple[str, ...]
    shell: str
    source: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    preflight: tuple[CheckCommand, ...]
    preflight_after_setup: tuple[CheckCommand, ...]
    run_prefix: tuple[str, ...]


def default_profiles_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "launch_profiles.json"


def user_profiles_path() -> Path:
    return Path(os.environ.get("UCL_MACHINE_TOOLS_CONFIG", "~/.config/ucl-machine-tools")).expanduser() / "launch_profiles.json"


def validate_name(value: str, label: str = "name") -> None:
    if not value or not SAFE_NAME_RE.match(value):
        raise ValueError(f"{label} may only contain letters, numbers, dot, dash, and underscore: {value!r}")


def validate_env_key(key: str) -> None:
    if not ENV_KEY_RE.match(key):
        raise ValueError(f"invalid env key: {key!r}")


def _reject_control_text(value: str, label: str) -> None:
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL bytes")


def _string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, list):
        items = tuple(value)
    else:
        raise ValueError(f"{label} must be a string or list of strings")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} entries must be non-empty strings")
        _reject_control_text(item, label)
        out.append(item)
    return tuple(out)


def _check_commands(value: Any, *, label: str) -> tuple[CheckCommand, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    out: list[CheckCommand] = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"{label}[{idx}] must be an object")
        extra = set(raw) - {"label", "command"}
        if extra:
            raise ValueError(f"{label}[{idx}] contains unknown fields: {', '.join(sorted(extra))}")
        check_label = raw.get("label")
        command = raw.get("command")
        if not isinstance(check_label, str) or not check_label:
            raise ValueError(f"{label}[{idx}].label must be a non-empty string")
        if not isinstance(command, str) or not command:
            raise ValueError(f"{label}[{idx}].command must be a non-empty string")
        _reject_control_text(check_label, f"{label}[{idx}].label")
        _reject_control_text(command, f"{label}[{idx}].command")
        out.append(CheckCommand(check_label, command))
    return tuple(out)


def _env_items(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("env must be an object")
    out: list[tuple[str, str]] = []
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise ValueError("env keys must be strings")
        validate_env_key(key)
        if not isinstance(raw_value, str):
            raise ValueError(f"env value for {key!r} must be a string")
        _reject_control_text(raw_value, f"env {key}")
        out.append((key, raw_value))
    return tuple(out)


def _parse_profile(name: str, raw: Any) -> ProfileDef:
    validate_name(name, "profile name")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"profile {name!r} must be an object")
    extra = set(raw) - ALLOWED_FIELDS
    if extra:
        raise ValueError(f"profile {name!r} contains unknown fields: {', '.join(sorted(extra))}")
    shell = raw.get("shell")
    if shell is not None:
        if not isinstance(shell, str) or shell not in ALLOWED_SHELLS:
            raise ValueError(f"profile {name!r} has invalid shell: {shell!r}")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"profile {name!r} description must be a string")
    source = _string_list(raw.get("source"), label=f"profile {name!r} source")
    for path in source:
        if not path.startswith("/"):
            raise ValueError(f"profile {name!r} source paths must be absolute: {path!r}")
        if "\n" in path or "\r" in path:
            raise ValueError(f"profile {name!r} source path must be one line: {path!r}")
    run_prefix = _string_list(raw.get("run_prefix"), label=f"profile {name!r} run_prefix")
    extends = _string_list(raw.get("extends"), label=f"profile {name!r} extends")
    for parent in extends:
        validate_name(parent, "extends")
    return ProfileDef(
        name=name,
        description=description,
        extends=extends,
        shell=shell,
        source=source,
        env=_env_items(raw.get("env")),
        preflight=_check_commands(raw.get("preflight"), label=f"profile {name!r} preflight"),
        preflight_after_setup=_check_commands(
            raw.get("preflight_after_setup"),
            label=f"profile {name!r} preflight_after_setup",
        ),
        run_prefix=run_prefix,
    )


def load_profile_file(path: Path | str) -> dict[str, ProfileDef]:
    profile_path = Path(path).expanduser()
    with profile_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"profile file must contain an object: {profile_path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"profile file {profile_path} has unsupported schema_version: {data.get('schema_version')!r}")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"profile file {profile_path} must contain a profiles object")
    return {str(name): _parse_profile(str(name), raw) for name, raw in profiles.items()}


def load_profiles(*, explicit_files: Iterable[Path | str] = ()) -> dict[str, ProfileDef]:
    merged: dict[str, ProfileDef] = {}
    for path in (default_profiles_path(), user_profiles_path(), *tuple(explicit_files)):
        profile_path = Path(path).expanduser()
        if profile_path == user_profiles_path() and not profile_path.exists():
            continue
        loaded = load_profile_file(profile_path)
        merged.update(loaded)
    return merged


def _merge_env(base: tuple[tuple[str, str], ...], new: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    merged = dict(base)
    for key, value in new:
        merged[key] = value
    return tuple(merged.items())


def resolve_profiles(
    selected: Iterable[str],
    *,
    catalog: dict[str, ProfileDef],
    cli_env: Iterable[tuple[str, str]] = (),
) -> ResolvedProfile:
    names = tuple(selected) or ("plain-bash",)
    for name in names:
        validate_name(name, "profile")

    resolved_defs: list[ProfileDef] = []
    visiting: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            cycle = " -> ".join((*visiting, name))
            raise ValueError(f"profile extends cycle: {cycle}")
        profile = catalog.get(name)
        if profile is None:
            raise ValueError(f"unknown profile: {name}")
        visiting.append(name)
        for parent in profile.extends:
            visit(parent)
        visiting.pop()
        if name not in seen:
            resolved_defs.append(profile)
            seen.add(name)

    for name in names:
        visit(name)

    shell = "bash"
    sources: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    preflight: tuple[CheckCommand, ...] = ()
    after: tuple[CheckCommand, ...] = ()
    run_prefix: tuple[str, ...] = ()
    for profile in resolved_defs:
        if profile.shell is not None:
            if shell != "bash" and shell != profile.shell:
                raise ValueError(f"conflicting profile shells: {shell!r} and {profile.shell!r}")
            shell = profile.shell
        sources = (*sources, *profile.source)
        env = _merge_env(env, profile.env)
        preflight = (*preflight, *profile.preflight)
        after = (*after, *profile.preflight_after_setup)
        if profile.run_prefix:
            if run_prefix and run_prefix != profile.run_prefix:
                raise ValueError("multiple conflicting run_prefix declarations")
            run_prefix = profile.run_prefix

    env = _merge_env(env, tuple(cli_env))
    if sources and shell != "csh-bootstrap":
        raise ValueError("source entries require shell='csh-bootstrap'")
    return ResolvedProfile(
        names=names,
        shell=shell,
        source=sources,
        env=env,
        preflight=preflight,
        preflight_after_setup=after,
        run_prefix=run_prefix,
    )


def parse_cli_env(items: Iterable[str]) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"env must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        validate_env_key(key)
        _reject_control_text(value, f"env {key}")
        parsed.append((key, value))
    return tuple(parsed)


def shell_join(tokens: Iterable[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def bash_export_lines(env: Iterable[tuple[str, str]]) -> list[str]:
    return [f"export {key}={shlex.quote(value)}" for key, value in env]


def csh_setenv_lines(env: Iterable[tuple[str, str]]) -> list[str]:
    return [f"setenv {key} {shlex.quote(value)}" for key, value in env]
