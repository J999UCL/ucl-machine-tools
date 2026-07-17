from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl_machine_tools import stage_registry
from ucl_machine_tools.stage_registry import (
    StageRecord,
    list_records,
    read_record,
    registry_root,
    update_status,
    write_record,
)


def make_record(
    stage_id: str = "fpt-barbury-a1b2c3d4",
    *,
    provenance: dict[str, Any] | None = None,
    status: str = "preparing",
) -> StageRecord:
    root = "/tmp/thakwani/fpt"
    return StageRecord(
        stage_id=stage_id,
        name="fpt",
        host="barbury-l",
        ssh_host="barbury-l.cs.ucl.ac.uk",
        remote_root=root,
        source_path=f"{root}/stages/fpt/sources/{'a' * 64}",
        environment_path=f"{root}/stages/fpt/envs/{'c' * 64}",
        uv_path=f"{root}/tools/uv/1.2.3/uv",
        cache_path=f"{root}/cache/uv",
        python_path=f"{root}/tools/python",
        state_path=f"{root}/stages/fpt/state/{stage_id}.json",
        source_hash="a" * 64,
        lock_hash="b" * 64,
        setup_environment_hash="d" * 64,
        uv_version="1.2.3",
        python_request="3.11.5",
        setup_run_id=f"{stage_id}-setup",
        status=status,
        provenance=provenance or {},
    )


def test_registry_root_uses_cache_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path))

    assert registry_root() == tmp_path / "stages"


def test_registry_root_uses_default_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UCL_MACHINE_TOOLS_CACHE", raising=False)

    assert registry_root() == Path("~/.cache/ucl-machine-tools/stages").expanduser()


def test_write_read_latest_and_list_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter(("2026-07-17T10:00:00Z", "2026-07-17T11:00:00Z"))
    monkeypatch.setattr(stage_registry, "utc_now", lambda: next(times))

    first = write_record(make_record("first"), root=tmp_path)
    second = write_record(make_record("second"), root=tmp_path)

    assert first.created_at == "2026-07-17T10:00:00Z"
    assert first.updated_at == "2026-07-17T10:00:00Z"
    assert second.created_at == "2026-07-17T11:00:00Z"
    assert read_record("first", root=tmp_path) == first
    assert read_record("last", root=tmp_path) == second
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == {
        "stage_id": "second"
    }
    assert [record.stage_id for record in list_records(root=tmp_path)] == ["first", "second"]


def test_write_preserves_existing_created_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter(("2026-07-17T10:00:00Z", "2026-07-17T11:00:00Z"))
    monkeypatch.setattr(stage_registry, "utc_now", lambda: next(times))
    original = write_record(make_record(), root=tmp_path)

    rewritten = write_record(original, root=tmp_path)

    assert rewritten.created_at == "2026-07-17T10:00:00Z"
    assert rewritten.updated_at == "2026-07-17T11:00:00Z"


def test_write_does_not_mutate_provenance_and_redacts_secrets(tmp_path: Path) -> None:
    provenance = {
        "project": "fpt",
        "env": {"HF_TOKEN": "hf-secret", "MODE": "train"},
        "api_token": "token-secret",
        "nested": {"password": "password-secret", "safe": "kept"},
    }
    expected_caller_value = json.loads(json.dumps(provenance))

    persisted = write_record(make_record(provenance=provenance), root=tmp_path)
    raw = (tmp_path / f"{persisted.stage_id}.json").read_text(encoding="utf-8")

    assert provenance == expected_caller_value
    assert persisted.provenance == {
        "project": "fpt",
        "env": {"HF_TOKEN": "<redacted>", "MODE": "<redacted>"},
        "api_token": "<redacted>",
        "nested": {"password": "<redacted>", "safe": "kept"},
    }
    assert "hf-secret" not in raw
    assert "token-secret" not in raw
    assert "password-secret" not in raw


def test_read_backfills_additive_fields_for_legacy_record(tmp_path: Path) -> None:
    record = make_record("legacy")
    payload = {
        key: value
        for key, value in record.__dict__.items()
        if key
        not in {
            "python_path",
            "state_path",
            "setup_run_id",
            "status",
            "created_at",
            "updated_at",
            "provenance",
        }
    }
    (tmp_path / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = read_record("legacy", root=tmp_path)

    assert loaded.python_path == ""
    assert loaded.state_path == ""
    assert loaded.setup_run_id == ""
    assert loaded.status == "unknown"
    assert loaded.created_at == ""
    assert loaded.updated_at == ""
    assert loaded.provenance == {}


@pytest.mark.parametrize(
    ("ref", "contents", "message"),
    [
        ("missing", None, "stage record not found: missing"),
        ("broken", "{not-json", "invalid stage record JSON: broken"),
        ("wrong-shape", "[]", "invalid stage record wrong-shape: expected an object"),
        ("incomplete", '{"stage_id": "incomplete"}', "missing required field"),
    ],
)
def test_read_reports_missing_and_corrupt_records_clearly(
    tmp_path: Path,
    ref: str,
    contents: str | None,
    message: str,
) -> None:
    if contents is not None:
        (tmp_path / f"{ref}.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_record(ref, root=tmp_path)


def test_read_last_reports_invalid_and_missing_pointer_targets(tmp_path: Path) -> None:
    (tmp_path / "latest.json").write_text('{"no_stage_id": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid latest stage pointer"):
        read_record("last", root=tmp_path)

    (tmp_path / "latest.json").write_text('{"stage_id": "gone"}', encoding="utf-8")
    with pytest.raises(ValueError, match="latest stage record target not found: gone"):
        read_record("last", root=tmp_path)


def test_record_ids_cannot_escape_registry_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid stage id"):
        write_record(make_record("../escape"), root=tmp_path)
    with pytest.raises(ValueError, match="invalid stage id"):
        read_record("../escape", root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_hash", "bad", "source_hash"),
        ("status", "made_up", "status"),
        ("source_path", "/tmp/outside", "source_path"),
        ("uv_path", "relative/uv", "uv_path"),
        ("host", "bad host", "host"),
    ],
)
def test_registry_rejects_untrusted_execution_metadata(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    values = {**make_record().__dict__, field: value}
    with pytest.raises(ValueError, match=message):
        write_record(StageRecord(**values), root=tmp_path)


def test_atomic_write_preserves_record_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = write_record(make_record(status="preparing"), root=tmp_path)
    original_text = (tmp_path / f"{original.stage_id}.json").read_text(encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(stage_registry.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_record(make_record(status="ready"), root=tmp_path)

    assert (tmp_path / f"{original.stage_id}.json").read_text(encoding="utf-8") == original_text
    assert not list(tmp_path.glob("*.tmp"))


def test_update_status_returns_new_record_and_preserves_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter(("2026-07-17T10:00:00Z", "2026-07-17T10:05:00Z"))
    monkeypatch.setattr(stage_registry, "utc_now", lambda: next(times))
    original = write_record(make_record(provenance={"phase": "upload"}), root=tmp_path)

    updated = update_status(
        original.stage_id,
        "ready",
        setup_run_id="setup-finished",
        provenance={"reused": True},
        root=tmp_path,
    )

    assert original.status == "preparing"
    assert original.setup_run_id.endswith("-setup")
    assert original.provenance == {"phase": "upload"}
    assert updated.status == "ready"
    assert updated.setup_run_id == "setup-finished"
    assert updated.provenance == {"phase": "upload", "reused": True}
    assert updated.created_at == original.created_at
    assert updated.updated_at == "2026-07-17T10:05:00Z"
    assert read_record(original.stage_id, root=tmp_path) == updated


def test_list_records_skips_corrupt_files_and_sorts_by_time(tmp_path: Path) -> None:
    older = make_record(
        "older",
        status="ready",
    )
    newer = make_record(
        "newer",
        status="ready",
    )
    write_record(
        StageRecord(**{**older.__dict__, "created_at": "2026-07-17T09:00:00Z"}),
        root=tmp_path,
    )
    write_record(
        StageRecord(**{**newer.__dict__, "created_at": "2026-07-17T10:00:00Z"}),
        root=tmp_path,
    )
    (tmp_path / "corrupt.json").write_text("not JSON", encoding="utf-8")

    assert [record.stage_id for record in list_records(root=tmp_path)] == ["older", "newer"]
