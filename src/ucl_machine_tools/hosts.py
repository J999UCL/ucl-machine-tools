"""Static host catalog and selector resolution for remote GPU inventory."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SAFE_SSH_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class HostSpec:
    name: str
    ssh_host: str
    labels: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    scratch_root: str = "/tmp/ucl-machine-tools"
    restart_policy: str = "unknown"
    expected_gpu_count: int | None = None
    expected_gpu_name: str | None = None
    warning: str = ""


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "ucl_hosts.json"


def _validate_token(value: str, label: str) -> None:
    if not value or not _SAFE_TOKEN_RE.match(value):
        raise ValueError(f"{label} contains unsafe characters: {value!r}")


def _validate_ssh_host(value: str) -> None:
    if not value or not _SAFE_SSH_RE.match(value):
        raise ValueError(f"ssh_host contains unsafe characters: {value!r}")


def _validate_warning(value: str) -> None:
    if len(value) > 200 or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("host warning must be at most 200 printable characters")


def _catalog_specs_from_yaml(path: Path | str | None = None) -> list[HostSpec]:
    with Path(path or default_catalog_path()).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("host catalog must contain a mapping")
    defaults = data.get("defaults") or {}
    default_scratch = str(defaults.get("scratch_root", "/tmp/ucl-machine-tools"))
    raw_groups = data.get("groups") or {}
    raw_hosts = data.get("hosts") or {}
    if not isinstance(raw_groups, dict):
        raise ValueError("groups must be a mapping")
    if not isinstance(raw_hosts, dict) or not raw_hosts:
        raise ValueError("hosts must be a non-empty mapping")

    host_to_groups: dict[str, set[str]] = {str(name): set() for name in raw_hosts}
    for group, members in raw_groups.items():
        if not isinstance(members, list):
            raise ValueError(f"group {group!r} must be a list")
        for member in members:
            member_name = str(member)
            if member_name not in raw_hosts:
                raise ValueError(f"group {group!r} references unknown host {member_name!r}")
            host_to_groups[member_name].add(str(group))

    specs: list[HostSpec] = []
    for name, raw in raw_hosts.items():
        if not isinstance(raw, dict):
            raise ValueError(f"host {name!r} must be a mapping")
        host_name = str(name)
        gpu_class = str(raw.get("gpu_class", "unknown"))
        labels = {
            "ucl",
            "ucl-gpu",
            "cuda",
            gpu_class,
            *host_to_groups.get(host_name, set()),
        }
        labels.update(str(label) for label in raw.get("labels", ()) or ())
        aliases = tuple(str(alias) for alias in raw.get("aliases", ()) or ())
        specs.append(
            HostSpec(
                name=host_name,
                ssh_host=str(raw.get("ssh", host_name)),
                labels=tuple(sorted(labels)),
                aliases=aliases,
                scratch_root=str(raw.get("scratch_root", default_scratch)),
                restart_policy=str(raw.get("restart_policy", "unknown")),
                expected_gpu_count=int(raw["expected_gpu_count"]) if raw.get("expected_gpu_count") is not None else None,
                expected_gpu_name=str(raw["expected_gpu_name"]) if raw.get("expected_gpu_name") is not None else None,
                warning=str(raw.get("warning", "")),
            )
        )
    return specs


def validate_catalog(specs: Iterable[HostSpec]) -> dict[str, HostSpec]:
    catalog: dict[str, HostSpec] = {}
    seen_ssh: dict[str, str] = {}
    seen_aliases: dict[str, str] = {}
    for spec in specs:
        _validate_token(spec.name, "name")
        _validate_ssh_host(spec.ssh_host)
        if spec.name in catalog:
            raise ValueError(f"duplicate host name: {spec.name}")
        previous = seen_ssh.get(spec.ssh_host)
        if previous is not None and previous != spec.name:
            raise ValueError(f"duplicate ssh_host {spec.ssh_host!r} for {previous!r} and {spec.name!r}")
        if not spec.scratch_root.startswith("/"):
            raise ValueError(f"scratch_root must be absolute for {spec.name}: {spec.scratch_root!r}")
        _validate_token(spec.restart_policy, "restart_policy")
        _validate_warning(spec.warning)
        for label in spec.labels:
            _validate_token(label, "label")
        for alias in spec.aliases:
            _validate_token(alias, "alias")
        for token in (spec.name, spec.ssh_host, *spec.aliases):
            previous_alias = seen_aliases.get(token)
            if previous_alias is not None and previous_alias != spec.name:
                raise ValueError(f"duplicate alias {token!r} for {previous_alias!r} and {spec.name!r}")
            seen_aliases[token] = spec.name
        catalog[spec.name] = spec
        seen_ssh[spec.ssh_host] = spec.name
    return catalog


def load_catalog(path: Path | str | None = None) -> dict[str, HostSpec]:
    return validate_catalog(_catalog_specs_from_yaml(path))


DEFAULT_CATALOG = tuple(_catalog_specs_from_yaml())


def parse_selector(selector: str, *, catalog: dict[str, HostSpec] | None = None) -> list[HostSpec]:
    active_catalog = catalog if catalog is not None else validate_catalog(DEFAULT_CATALOG)
    if selector is None or not selector.strip():
        raise ValueError("selector must be non-empty")

    included: set[str] = set()
    excluded: set[str] = set()

    def matching_names(token: str) -> list[str]:
        if token == "all":
            return [
                name for name, spec in active_catalog.items() if "restricted" not in spec.labels
            ]
        if token.startswith("label:"):
            label = token.split(":", 1)[1]
            if not label:
                raise ValueError("label selector must be non-empty")
            return [
                name
                for name, spec in active_catalog.items()
                if label in spec.labels and "restricted" not in spec.labels
            ]
        if token in active_catalog:
            return [token]
        alias_matches = [name for name, spec in active_catalog.items() if token == spec.ssh_host or token in spec.aliases]
        if alias_matches:
            return alias_matches
        label_matches = [
            name
            for name, spec in active_catalog.items()
            if token in spec.labels and "restricted" not in spec.labels
        ]
        if label_matches:
            return label_matches
        raise ValueError(f"unknown selector token: {token}")

    for raw_token in selector.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("selector contains an empty token")
        is_exclusion = token.startswith("!")
        token = token[1:] if is_exclusion else token
        _validate_token(token, "selector")
        matches = matching_names(token)
        if not matches:
            raise ValueError(f"selector token matched no hosts: {token}")
        if is_exclusion:
            excluded.update(matches)
        else:
            included.update(matches)

    selected = [spec for name, spec in active_catalog.items() if name in included and name not in excluded]
    if not selected:
        raise ValueError(f"selector matched no hosts after exclusions: {selector!r}")
    return selected
