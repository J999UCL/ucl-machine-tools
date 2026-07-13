from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl_machine_tools import registry
from ucl_machine_tools.registry import RunRecord, list_records, read_record, write_record


def make_record(run_id: str, *, identity: dict[str, Any] | None = None) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        kind="run",
        host="barbury-l",
        ssh_host="barbury-l",
        session=run_id,
        window="run",
        remote_dir=f"/tmp/ucl-machine-tools/launchers/{run_id}",
        log_path=f"/tmp/ucl-machine-tools/launchers/{run_id}/run.log",
        command=("bash", "run.sh"),
        identity=identity or {},
    )


def test_read_record_backfills_identity_for_legacy_record(tmp_path: Path) -> None:
    payload = {
        "run_id": "legacy",
        "kind": "run",
        "host": "barbury-l",
        "ssh_host": "barbury-l",
        "session": "legacy",
        "window": "run",
        "remote_dir": "/tmp/ucl-machine-tools/launchers/legacy",
        "log_path": "/tmp/ucl-machine-tools/launchers/legacy/run.log",
        "command": ["bash", "run.sh"],
    }
    (tmp_path / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")

    record = read_record("legacy", root=tmp_path)

    assert record.command == ("bash", "run.sh")
    assert record.identity == {}


def test_identity_round_trips_through_persisted_record(tmp_path: Path) -> None:
    identity = {
        "backend": "slurm",
        "job_id": "12345",
        "array": {"task_id": 7},
    }

    write_record(make_record("identified", identity=identity), root=tmp_path)

    payload = json.loads((tmp_path / "identified.json").read_text(encoding="utf-8"))
    assert "identity" not in payload
    assert payload["provenance"]["job_identity"] == identity
    assert read_record("identified", root=tmp_path).identity == identity


def test_atomic_record_write_preserves_existing_json_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_record(make_record("stable", identity={"generation": 1}), root=tmp_path)
    original = (tmp_path / "stable.json").read_text(encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(registry.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_record(make_record("stable", identity={"generation": 2}), root=tmp_path)

    assert (tmp_path / "stable.json").read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.tmp"))


def test_read_record_supports_legacy_top_level_identity(tmp_path: Path) -> None:
    payload = {
        "run_id": "legacy-identity",
        "kind": "run",
        "host": "barbury-l",
        "ssh_host": "barbury-l",
        "session": "legacy-identity",
        "window": "run",
        "remote_dir": "/tmp/ucl-machine-tools/launchers/legacy-identity",
        "log_path": "/tmp/ucl-machine-tools/launchers/legacy-identity/run.log",
        "command": ["bash", "run.sh"],
        "provenance": {"project": "example"},
        "identity": {"pane_id": "%1"},
    }
    (tmp_path / "legacy-identity.json").write_text(json.dumps(payload), encoding="utf-8")

    record = read_record("legacy-identity", root=tmp_path)

    assert record.identity == {"pane_id": "%1"}
    assert record.provenance == {"project": "example"}


def test_read_record_prefers_legacy_identity_when_both_schemas_exist(tmp_path: Path) -> None:
    payload = {
        "run_id": "transitional",
        "kind": "run",
        "host": "barbury-l",
        "ssh_host": "barbury-l",
        "session": "transitional",
        "window": "run",
        "remote_dir": "/tmp/ucl-machine-tools/launchers/transitional",
        "log_path": "/tmp/ucl-machine-tools/launchers/transitional/run.log",
        "command": ["bash", "run.sh"],
        "provenance": {"job_identity": {"pane_id": "%new"}},
        "identity": {"pane_id": "%legacy"},
    }
    (tmp_path / "transitional.json").write_text(json.dumps(payload), encoding="utf-8")

    record = read_record("transitional", root=tmp_path)

    assert record.identity == {"pane_id": "%legacy"}


def test_latest_and_list_records_support_mixed_identity_schemas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_payload = {
        "run_id": "legacy",
        "kind": "run",
        "host": "barbury-l",
        "ssh_host": "barbury-l",
        "session": "legacy",
        "window": "run",
        "remote_dir": "/tmp/ucl-machine-tools/launchers/legacy",
        "log_path": "/tmp/ucl-machine-tools/launchers/legacy/run.log",
        "command": ["bash", "run.sh"],
        "created_at": "2026-07-13T10:00:00Z",
        "updated_at": "2026-07-13T10:00:00Z",
    }
    (tmp_path / "legacy.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
    monkeypatch.setattr(registry, "utc_now", lambda: "2026-07-13T11:00:00Z")
    write_record(make_record("current", identity={"job_id": "67890"}), root=tmp_path)

    latest = read_record(root=tmp_path)
    records = list_records(root=tmp_path)

    assert not (tmp_path / "latest.json").is_symlink()
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))["run_id"] == "current"
    assert latest.run_id == "current"
    assert latest.identity == {"job_id": "67890"}
    assert [record.run_id for record in records] == ["legacy", "current"]
    assert [record.identity for record in records] == [{}, {"job_id": "67890"}]


def test_read_record_supports_legacy_full_latest_file(tmp_path: Path) -> None:
    record = make_record("legacy-latest")
    payload = {
        "run_id": record.run_id,
        "kind": record.kind,
        "host": record.host,
        "ssh_host": record.ssh_host,
        "session": record.session,
        "window": record.window,
        "remote_dir": record.remote_dir,
        "log_path": record.log_path,
        "command": list(record.command),
    }
    (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

    assert read_record("last", root=tmp_path).run_id == "legacy-latest"
