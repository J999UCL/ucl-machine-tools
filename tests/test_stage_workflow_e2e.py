from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import envcheck, job_control, launch, main_cli, stage as stage_tools, stage_registry, uv_remote
from ucl_machine_tools.registry import list_records as list_run_records
from ucl_machine_tools.stage_registry import StageRecord
from ucl_machine_tools.uv_project import build_source_manifest


UV_VERSION = "0.9.27"
REMOTE_ROOT = "/tmp/thakwani/demo"


def completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "defaults": {"scratch_root": "/tmp/ucl-machine-tools"},
                "groups": {"3090ti": ["barbury-l"]},
                "hosts": {
                    "barbury-l": {
                        "gpu_class": "3090ti",
                        "restart_policy": "lab_pc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return catalog


def write_locked_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """\
[project]
name = "stage-e2e"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text(
        """\
version = 1
revision = 3
requires-python = ">=3.11"

[[package]]
name = "stage-e2e"
version = "0.1.0"
source = { virtual = "." }
""",
        encoding="utf-8",
    )
    (project / ".python-version").write_text("3.11\n", encoding="utf-8")
    scripts = project / "scripts"
    scripts.mkdir()
    run_script = scripts / "run.sh"
    run_script.write_text("#!/usr/bin/env bash\npython3 -c 'print(42)'\n", encoding="utf-8")
    run_script.chmod(0o755)
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    # Mandatory staging exclusions must not affect the source identity or transfer.
    (project / "data").mkdir()
    (project / "data" / "large.bin").write_bytes(b"do-not-upload")
    (project / ".venv").mkdir()
    (project / ".venv" / "python").write_text("not portable\n", encoding="utf-8")
    (project / ".env").write_text("HF_TOKEN=secret\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    uv.chmod(0o755)
    return project, uv


def tmux_stdout(sessions: tuple[str, ...] = ("unrelated",)) -> str:
    return "\n".join(
        (
            launch.TMUX_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "sessions": list(sessions)}),
            launch.TMUX_SENTINEL_END,
        )
    )


def launch_stdout(*, session: str, window: str) -> str:
    identity = {
        "exists": True,
        "session": session,
        "window": window,
        "boot_id": "boot-e2e",
        "tmux_socket_path": "/tmp/tmux-e2e/default",
        "tmux_server_pid": 4000,
        "pane_id": "%9",
        "window_id": "@4",
        "pane_pid": 4100,
        "pane_start_ticks": 12345,
        "pane_session_id": 4100,
        "pane_dead": False,
        "pane_dead_status": None,
    }
    return "\n".join(
        (
            job_control.LAUNCH_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "ok": True, "identity": identity, "error": ""}),
            job_control.LAUNCH_SENTINEL_END,
        )
    )


def setup_state(record: StageRecord, *, status: str = "ready") -> dict[str, Any]:
    ready = status == "ready"
    return {
        "schema_version": uv_remote.SCHEMA_VERSION,
        "ok": ready,
        "status": status,
        "phase": "ready" if ready else "sync",
        "uv_version": record.uv_version,
        "source_sha256": record.source_hash,
        "lock_sha256": record.lock_hash,
        "python_request": record.python_request,
        "python_path": record.python_path,
        "source_dir": record.source_path,
        "environment_dir": record.environment_path,
        "uv_binary_path": record.uv_path,
        "ready_state_path": record.state_path,
        "failed_state_path": record.state_path.replace(".json", ".failed.json"),
        "log_path": f"{record.remote_root}/launchers/{record.setup_run_id}/setup.log",
        "reused_uv": False,
        "reused_environment": False,
        "returncode": 0 if ready else 1,
        "error": "" if ready else "dependency installation failed",
        "failed_command": "" if ready else "uv sync --frozen --no-editable",
        "failed_line": None if ready else 91,
    }


class FakePopen:
    def __init__(self, owner: "FakeUclRunner", argv: list[str], **kwargs: Any) -> None:
        self.owner = owner
        self.argv = tuple(str(token) for token in argv)
        self.kwargs = kwargs
        self.returncode = 0
        self.stdout = SimpleNamespace(close=lambda: None)
        owner.popen_calls.append((self.argv, kwargs))

    def wait(self) -> int:
        return self.returncode

    def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


class FakeUclRunner:
    """Stateful subprocess boundary for CLI-level staging tests.

    Unknown remote probes deliberately return successful empty output. Specific
    structured operations receive the same sentinels used by the real helpers.
    This keeps the suite focused on workflow semantics rather than SSH script
    formatting.
    """

    def __init__(self, uv_path: Path) -> None:
        self.uv_path = str(uv_path.resolve())
        self.lock_error = ""
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.popen_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.remote_state: dict[str, Any] | None = None
        self.remote_state_missing = False
        self.tmux_sessions: tuple[str, ...] = ("unrelated",)
        self.ready_sources: set[str] = set()

    def popen(self, argv: list[str], **kwargs: Any) -> FakePopen:
        return FakePopen(self, argv, **kwargs)

    def __call__(self, argv: list[str], **kwargs: Any) -> SimpleNamespace:
        tokens = tuple(str(token) for token in argv)
        self.calls.append((tokens, dict(kwargs)))

        if tokens == ("ssh", "-O", "check", "knuckles"):
            return completed()
        if tokens and os.path.realpath(tokens[0]) == self.uv_path:
            if tokens[1:] == ("--version",):
                return completed(stdout=f"uv {UV_VERSION}\n")
            if tokens[1:3] == ("lock", "--check"):
                if self.lock_error:
                    return completed(returncode=1, stderr=self.lock_error)
                return completed()
            return completed(returncode=97, stderr=f"unexpected local uv command: {tokens!r}")
        if tokens and tokens[0] == "rsync":
            return completed()
        if not tokens or tokens[0] not in {"ssh", "python3"}:
            return completed()

        source = kwargs.get("input")
        source_text = source.decode() if isinstance(source, bytes) else str(source or "")
        if envcheck.ENV_BEGIN in source_text:
            payload = {
                "schema_version": 1,
                "remote_root": REMOTE_ROOT,
                "root_exists": False,
                "root_created": False,
                "tmp_free_gb": 600.0,
                "cuda_visibility_script": "/usr/local/cuda/CUDA_VISIBILITY.csh",
                "cuda_visibility_exists": True,
                "python_setup_script": "/opt/Python/Python-3.11.5_Setup.csh",
                "python_setup_exists": True,
                "gpu": None,
                "gpu_info": None,
                "ok": True,
                "errors": [],
            }
            return completed(
                stdout="\n".join((envcheck.ENV_BEGIN, json.dumps(payload), envcheck.ENV_END))
            )
        if stage_tools.SOURCE_SENTINEL_BEGIN in source_text:
            if '"ready": False' in source_text:
                source_dir = _assigned_python_value(source_text, "SOURCE", path_wrapper=True) or ""
                ready = source_dir in self.ready_sources
                payload = {"schema_version": 1, "ok": True, "ready": ready, "exists": ready, "error": ""}
            else:
                source_dir = _assigned_python_value(source_text, "FINAL", path_wrapper=True) or "/tmp/source"
                expected_text = _assigned_python_value(source_text, "EXPECTED", json_loads=True) or "{}"
                expected = json.loads(expected_text)
                self.ready_sources.add(source_dir)
                payload = {
                    "schema_version": 1,
                    "ok": True,
                    "reused": False,
                    "source_dir": source_dir,
                    "source_sha256": expected["source_sha256"],
                    "file_count": expected["file_count"],
                    "total_bytes": expected["total_bytes"],
                }
            return completed(
                stdout="\n".join(
                    (stage_tools.SOURCE_SENTINEL_BEGIN, json.dumps(payload), stage_tools.SOURCE_SENTINEL_END)
                )
            )
        if stage_tools.STATE_SENTINEL_BEGIN in source_text:
            if self.remote_state_missing or self.remote_state is None:
                payload = {"schema_version": 1, "status": "missing", "state": None}
            else:
                payload = {
                    "schema_version": 1,
                    "status": self.remote_state["status"],
                    "state": self.remote_state,
                    "missing_paths": [],
                }
            return completed(
                stdout="\n".join(
                    (stage_tools.STATE_SENTINEL_BEGIN, json.dumps(payload), stage_tools.STATE_SENTINEL_END)
                )
            )
        if launch.TMUX_SENTINEL_BEGIN in source_text:
            return completed(stdout=tmux_stdout(self.tmux_sessions))
        if job_control.LAUNCH_SENTINEL_BEGIN in source_text:
            session = _assigned_json_string(source_text, "EXPECTED_SESSION") or "stage-e2e"
            window = _assigned_json_string(source_text, "EXPECTED_WINDOW") or "setup"
            return completed(stdout=launch_stdout(session=session, window=window))

        return completed()

    def joined_calls(self, *, start: int = 0) -> str:
        chunks: list[str] = []
        for argv, kwargs in self.calls[start:]:
            chunks.append(" ".join(argv))
            value = kwargs.get("input")
            if isinstance(value, bytes):
                chunks.append(value.decode("utf-8", errors="replace"))
            elif isinstance(value, str):
                chunks.append(value)
        for argv, _ in self.popen_calls:
            chunks.append(" ".join(argv))
        return "\n".join(chunks)

    def transfer_count(self) -> int:
        def is_transfer(argv: tuple[str, ...]) -> bool:
            return bool(argv) and (
                argv[0] == "rsync"
                or argv[:2] == ("tar", "-cf")
                or (argv[0] == "ssh" and any("tar -xf" in token for token in argv))
            )

        return sum(is_transfer(argv) for argv, _ in self.calls) + sum(
            is_transfer(argv) for argv, _ in self.popen_calls
        )


def _assigned_json_string(source: str, name: str) -> str | None:
    prefix = f"{name}="
    for line in source.splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, str) else None
    return None


def _assigned_python_value(
    source: str,
    name: str,
    *,
    path_wrapper: bool = False,
    json_loads: bool = False,
) -> str | None:
    import ast

    prefix = f"{name} = "
    for line in source.splitlines():
        if not line.startswith(prefix):
            continue
        expression = line[len(prefix) :]
        if path_wrapper and expression.startswith("Path(") and expression.endswith(")"):
            expression = expression[5:-1]
        if json_loads and expression.startswith("json.loads(") and expression.endswith(")"):
            expression = expression[11:-1]
        value = ast.literal_eval(expression)
        return value if isinstance(value, str) else None
    return None


def _looks_like_state_probe(source: str) -> bool:
    lowered = source.lower()
    return "state" in lowered and ("read_text" in lowered or "json.load" in lowered or "cat" in lowered)


@pytest.fixture
def workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    project, uv_path = write_locked_project(tmp_path)
    catalog = write_catalog(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(cache))
    monkeypatch.setenv("PATH", f"{uv_path.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    runner = FakeUclRunner(uv_path)
    return SimpleNamespace(
        tmp_path=tmp_path,
        project=project,
        catalog=catalog,
        cache=cache,
        runner=runner,
    )


def stage_argv(workflow: SimpleNamespace, *extra: str) -> list[str]:
    return [
        "stage",
        "--uv",
        "--host",
        "barbury-l",
        "--catalog",
        str(workflow.catalog),
        "--name",
        "demo",
        "--local-dir",
        str(workflow.project),
        "--remote-root",
        REMOTE_ROOT,
        *extra,
    ]


def invoke_stage(workflow: SimpleNamespace, *extra: str) -> int:
    return main_cli.main(
        stage_argv(workflow, *extra),
        runner=workflow.runner,
        popener=workflow.runner.popen,
    )


def test_stage_validates_locked_project_before_remote_mutation(
    workflow: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workflow.runner.lock_error = "The lockfile needs to be updated"

    rc = invoke_stage(workflow)

    captured = capsys.readouterr()
    assert rc == 2
    assert "lockfile needs to be updated" in captured.err.lower()
    assert workflow.runner.transfer_count() == 0
    assert not any(call[0] == "ssh" for call, _ in workflow.runner.calls)
    assert stage_registry.list_records() == []


def test_stage_uploads_content_addressed_source_and_launches_dedicated_setup_tmux(
    workflow: SimpleNamespace,
) -> None:
    expected_manifest = build_source_manifest(workflow.project)

    assert invoke_stage(workflow) == 0

    records = stage_registry.list_records()
    assert len(records) == 1
    record = records[0]
    assert record.host == "barbury-l"
    assert record.remote_root == REMOTE_ROOT
    assert record.source_hash == expected_manifest.source_sha256
    assert record.source_path.endswith(f"/stages/demo/sources/{expected_manifest.source_sha256}")
    assert f"{REMOTE_ROOT}/stages/demo/envs/" in record.environment_path
    assert record.uv_version == UV_VERSION
    assert record.uv_path == f"{REMOTE_ROOT}/tools/uv/{UV_VERSION}/uv"
    assert record.cache_path == f"{REMOTE_ROOT}/cache/uv"
    assert record.state_path.endswith(f"/{record.stage_id}.json")
    assert record.setup_run_id
    assert record.status in {"preparing", "setting_up", "ready"}
    assert workflow.runner.transfer_count() >= 1

    all_calls = workflow.runner.joined_calls()
    assert "/opt/Python/Python-3.11.5_Setup.csh" in all_calls
    assert f"https://astral.sh/uv/{UV_VERSION}/install.sh" in all_calls
    assert "UV_UNMANAGED_INSTALL" in all_calls
    assert "sync --frozen --no-editable" in all_calls
    assert "sync --frozen --check" in all_calls
    assert "tmux" in all_calls and "new-session" in all_calls
    assert record.setup_run_id in all_calls

    manifest_paths = {entry.path for entry in expected_manifest.entries}
    assert "pyproject.toml" in manifest_paths
    assert "uv.lock" in manifest_paths
    assert ".python-version" in manifest_paths
    assert "scripts/run.sh" in manifest_paths
    assert not any(path.startswith("data/") for path in manifest_paths)
    assert not any(path.startswith(".venv/") for path in manifest_paths)
    assert ".env" not in manifest_paths


def test_identical_stage_reuses_content_addressed_source_and_environment(
    workflow: SimpleNamespace,
) -> None:
    assert invoke_stage(workflow) == 0
    first = stage_registry.list_records()[0]
    first_transfer_count = workflow.runner.transfer_count()

    # A change under a mandatory exclusion must not create a new source identity.
    (workflow.project / "data" / "large.bin").write_bytes(b"different ignored bytes")
    workflow.runner.remote_state = setup_state(replace(first, status="ready"))

    assert invoke_stage(workflow) == 0

    records = stage_registry.list_records()
    assert len(records) == 1
    second = records[0]
    assert second.stage_id == first.stage_id
    assert second.source_hash == first.source_hash
    assert second.lock_hash == first.lock_hash
    assert second.source_path == first.source_path
    assert second.environment_path == first.environment_path
    assert workflow.runner.transfer_count() == first_transfer_count


def test_included_source_change_creates_a_distinct_stage_identity(workflow: SimpleNamespace) -> None:
    assert invoke_stage(workflow) == 0
    first = stage_registry.list_records()[0]
    (workflow.project / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert invoke_stage(workflow) == 0

    records = stage_registry.list_records()
    assert len(records) == 2
    second = records[-1]
    assert second.stage_id != first.stage_id
    assert second.source_hash != first.source_hash
    assert second.source_path != first.source_path
    # Dependencies did not change, but environment identity remains bound to the
    # immutable source used for no-editable installation.
    assert second.environment_path != first.environment_path


def test_run_from_ready_stage_uses_frozen_no_sync_without_upload(
    workflow: SimpleNamespace,
) -> None:
    assert invoke_stage(workflow) == 0
    record = stage_registry.update_status(stage_registry.list_records()[0].stage_id, "ready")
    workflow.runner.remote_state = setup_state(record)
    calls_before = len(workflow.runner.calls)
    transfers_before = workflow.runner.transfer_count()

    rc = main_cli.main(
        [
            "run",
            "--stage",
            record.stage_id,
            "--catalog",
            str(workflow.catalog),
            "--script",
            "scripts/run.sh",
            "--arg",
            "hello world",
            "--new-session",
        ],
        runner=workflow.runner,
        popener=workflow.runner.popen,
    )

    assert rc == 0
    assert workflow.runner.transfer_count() == transfers_before
    new_calls = workflow.runner.joined_calls(start=calls_before)
    assert record.uv_path in new_calls
    assert "run --frozen --no-sync" in new_calls
    assert record.source_path in new_calls
    assert record.environment_path in new_calls
    assert "scripts/run.sh" in new_calls
    assert "hello world" in new_calls
    assert "sync --frozen" not in new_calls
    assert not any(argv and argv[0] == "rsync" for argv, _ in workflow.runner.calls[calls_before:])
    assert "tar -cf" not in new_calls

    runs = list_run_records()
    assert len(runs) >= 1
    staged_run = next(run for run in runs if run.provenance.get("stage_id") == record.stage_id and run.kind == "run")
    assert staged_run.host == record.host
    assert tuple(staged_run.command[:4]) == (record.uv_path, "run", "--frozen", "--no-sync")
    assert staged_run.provenance.get("stage_id") == record.stage_id
    assert staged_run.provenance.get("source_hash") == record.source_hash
    assert staged_run.provenance.get("lock_hash") == record.lock_hash


def test_run_rejects_unknown_stage_before_remote_work(
    workflow: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls_before = len(workflow.runner.calls)

    rc = main_cli.main(
        [
            "run",
            "--stage",
            "missing-stage",
            "--script",
            "scripts/run.sh",
            "--new-session",
        ],
        runner=workflow.runner,
        popener=workflow.runner.popen,
    )

    assert rc == 2
    assert "stage record not found" in capsys.readouterr().err.lower()
    assert len(workflow.runner.calls) == calls_before


@pytest.mark.parametrize("remote_status", ("failed", "missing"))
def test_run_rejects_failed_or_missing_remote_stage_state_without_launch(
    workflow: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    remote_status: str,
) -> None:
    assert invoke_stage(workflow) == 0
    record = stage_registry.update_status(stage_registry.list_records()[0].stage_id, "ready")
    if remote_status == "failed":
        workflow.runner.remote_state = setup_state(record, status="failed")
    else:
        workflow.runner.remote_state_missing = True
    calls_before = len(workflow.runner.calls)
    transfers_before = workflow.runner.transfer_count()

    rc = main_cli.main(
        [
            "run",
            "--stage",
            record.stage_id,
            "--script",
            "scripts/run.sh",
            "--new-session",
        ],
        runner=workflow.runner,
        popener=workflow.runner.popen,
    )

    assert rc == 2
    error = capsys.readouterr().err.lower()
    assert "stage" in error
    assert remote_status in error
    assert workflow.runner.transfer_count() == transfers_before
    new_calls = workflow.runner.joined_calls(start=calls_before)
    assert "tmux new-session" not in new_calls
    assert "run --frozen --no-sync" not in new_calls
