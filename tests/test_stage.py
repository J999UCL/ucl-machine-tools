from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ucl_machine_tools import stage
from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools.uv_project import build_source_manifest
from ucl_machine_tools.uv_remote import UvRemotePaths, UvSetupSpec, build_setup_payload


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / ".python-version").write_text("3.11.5\n", encoding="utf-8")
    (root / "src").mkdir()
    script = root / "src" / "run.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    script.chmod(0o755)
    (root / "src" / "alias.sh").symlink_to("run.sh")
    return root


def _materialize(manifest, destination: Path) -> None:
    destination.mkdir(parents=True)
    for entry in manifest.entries:
        source = manifest.root / entry.path
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.kind == "directory":
            target.mkdir(exist_ok=True)
        elif entry.kind == "symlink":
            target.symlink_to(entry.symlink_target)
        else:
            shutil.copy2(source, target)


def _run_generated(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )


def test_source_promoter_verifies_and_atomically_publishes_snapshot(tmp_path: Path) -> None:
    manifest = build_source_manifest(_project(tmp_path))
    sources = tmp_path / "remote" / "sources"
    incoming = sources / ".incoming"
    final = sources / manifest.source_sha256
    _materialize(manifest, incoming)

    process = _run_generated(
        stage.build_source_promote_source(
            manifest=manifest,
            incoming_dir=str(incoming),
            source_dir=str(final),
            sources_dir=str(sources),
        )
    )

    assert process.returncode == 0, process.stderr
    payload = stage._parse_sentinel(
        process.stdout,
        stage.SOURCE_SENTINEL_BEGIN,
        stage.SOURCE_SENTINEL_END,
        "source verification",
    )
    assert payload["ok"] is True
    assert payload["source_sha256"] == manifest.source_sha256
    assert final.is_dir()
    assert not incoming.exists()
    marker = json.loads((final / stage.SOURCE_MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["source_sha256"] == manifest.source_sha256
    assert (final / "src" / "run.sh").stat().st_mode & 0o111
    assert os.readlink(final / "src" / "alias.sh") == "run.sh"
    assert final.stat().st_mode & 0o222 == 0
    assert (final / "pyproject.toml").stat().st_mode & 0o222 == 0

    clean_probe = _run_generated(stage.build_source_probe_source(str(final), manifest))
    assert stage._parse_sentinel(
        clean_probe.stdout,
        stage.SOURCE_SENTINEL_BEGIN,
        stage.SOURCE_SENTINEL_END,
        "source probe",
    )["ready"] is True

    tracked = final / "pyproject.toml"
    tracked.chmod(0o644)
    tracked.write_text("tampered\n", encoding="utf-8")
    changed_probe = _run_generated(stage.build_source_probe_source(str(final), manifest))
    changed = stage._parse_sentinel(
        changed_probe.stdout,
        stage.SOURCE_SENTINEL_BEGIN,
        stage.SOURCE_SENTINEL_END,
        "source probe",
    )
    assert changed["ok"] is False
    assert any(word in str(changed["error"]) for word in ("differs", "writable"))


def test_source_promoter_rejects_changed_bytes_and_removes_incoming(tmp_path: Path) -> None:
    manifest = build_source_manifest(_project(tmp_path))
    sources = tmp_path / "remote" / "sources"
    incoming = sources / ".incoming"
    final = sources / manifest.source_sha256
    _materialize(manifest, incoming)
    (incoming / "src" / "run.sh").write_text("changed\n", encoding="utf-8")

    process = _run_generated(
        stage.build_source_promote_source(
            manifest=manifest,
            incoming_dir=str(incoming),
            source_dir=str(final),
            sources_dir=str(sources),
        )
    )

    assert process.returncode != 0
    payload = stage._parse_sentinel(
        process.stdout,
        stage.SOURCE_SENTINEL_BEGIN,
        stage.SOURCE_SENTINEL_END,
        "source verification",
    )
    assert payload["ok"] is False
    assert "mismatch" in str(payload["error"])
    assert not incoming.exists()
    assert not final.exists()


def test_source_probe_rejects_existing_unmanaged_directory(tmp_path: Path) -> None:
    manifest = build_source_manifest(_project(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    process = _run_generated(stage.build_source_probe_source(str(source), manifest))
    payload = stage._parse_sentinel(
        process.stdout,
        stage.SOURCE_SENTINEL_BEGIN,
        stage.SOURCE_SENTINEL_END,
        "source probe",
    )
    assert payload["ok"] is False
    assert "integrity verification" in str(payload["error"])


def test_failed_source_upload_requests_remote_incoming_cleanup(tmp_path: Path) -> None:
    manifest = build_source_manifest(_project(tmp_path))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        if len(calls) == 1:
            stdout = "\n".join(
                (
                    stage.SOURCE_SENTINEL_BEGIN,
                    json.dumps({"schema_version": 1, "ok": True, "ready": False, "exists": False, "error": ""}),
                    stage.SOURCE_SENTINEL_END,
                )
            )
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if len(calls) == 3:
            return SimpleNamespace(returncode=23, stdout="", stderr="partial transfer")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="partial transfer"):
        stage.sync_source_snapshot(
            HostSpec("barbury-l", "barbury-l"),
            manifest=manifest,
            source_dir=f"/tmp/sources/{manifest.source_sha256}",
            sources_dir="/tmp/sources",
            runner=runner,
        )

    assert len(calls) == 4
    cleanup_argv = calls[-1][0]
    assert "shutil.rmtree" in " ".join(cleanup_argv)
    assert ".incoming-" in " ".join(cleanup_argv)


def test_state_probe_reports_missing_managed_paths(tmp_path: Path) -> None:
    ready = tmp_path / "ready.json"
    failed = tmp_path / "failed.json"
    ready.write_text(
        json.dumps(
            {
                "source_dir": str(tmp_path / "missing-source"),
                "environment_dir": str(tmp_path / "missing-env"),
                "uv_binary_path": str(tmp_path / "missing-uv"),
                "python_path": str(tmp_path / "missing-python"),
            }
        ),
        encoding="utf-8",
    )
    process = _run_generated(stage.build_state_probe_source(str(ready), str(failed)))
    payload = stage._parse_sentinel(
        process.stdout,
        stage.STATE_SENTINEL_BEGIN,
        stage.STATE_SENTINEL_END,
        "state probe",
    )
    assert payload["status"] == "ready"
    assert payload["missing_paths"] == [
        "source_dir",
        "environment_dir",
        "uv_binary_path",
        "python_path",
    ]


def test_setup_payload_writer_uses_private_files_and_argv_only(tmp_path: Path) -> None:
    root = "/tmp/ucl-stage-test"
    paths = UvRemotePaths(
        source_dir=f"{root}/source",
        environment_dir=f"{root}/env",
        uv_cache_dir=f"{root}/cache/uv",
        uv_tool_dir=f"{root}/tools/uv/0.9.27",
        uv_binary_path=f"{root}/tools/uv/0.9.27/uv",
        python_install_dir=f"{root}/tools/python",
        ready_state_path=f"{root}/state/ready.json",
        failed_state_path=f"{root}/state/failed.json",
        log_path=f"{root}/setup/setup.log",
        environment_lock_path=f"{root}/state/env.lock",
        uv_tool_lock_path=f"{root}/state/uv.lock",
    )
    payload = build_setup_payload(
        UvSetupSpec(
            uv_version="0.9.27",
            paths=paths,
            source_sha256="a" * 64,
            lock_sha256="b" * 64,
            setup_environment_sha256="c" * 64,
            python_request="3.11.5",
        ),
        csh_driver_path=f"{root}/setup/setup.csh",
        bash_driver_path=f"{root}/setup/setup.sh",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    stage.write_setup_payload(HostSpec("barbury-l", "barbury-l"), payload, runner=runner)

    assert len(calls) == 2
    assert all(kwargs.get("shell") is False for _, kwargs in calls)
    assert all("chmod 700" in " ".join(argv) for argv, _ in calls)
    assert {kwargs["input"] for _, kwargs in calls} == {payload.csh_source, payload.bash_source}


@pytest.mark.parametrize("value", ("relative", "/", "/tmp/root/../escape"))
def test_stage_paths_reject_unsafe_values(tmp_path: Path, value: str) -> None:
    manifest = build_source_manifest(_project(tmp_path))
    with pytest.raises(ValueError):
        stage.build_source_probe_source(value, manifest)
