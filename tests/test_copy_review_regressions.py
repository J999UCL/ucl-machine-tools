from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import main_cli


def _ok() -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _unexpected_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
    raise AssertionError(f"unexpected command: {argv}")


def _write_file(root: Path, relative: str, data: bytes = b"payload") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _single_json_object(captured: pytest.CaptureResult[str]) -> dict[str, Any]:
    assert captured.err == ""
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(captured.out)
    assert captured.out[end:].strip() == ""
    assert isinstance(payload, dict)
    return payload


def _manifest_item(**overrides: Any) -> dict[str, Any]:
    data = b"same bytes"
    item = {
        "path": "payload.bin",
        "bytes": len(data),
        "kind": "file",
        "mode": 0o640,
        "mtime_ns": 1_700_000_000_000_000_000,
        "uid": 1000,
        "gid": 1000,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    item.update(overrides)
    return item


@pytest.mark.parametrize(
    ("metadata_field", "different_value"),
    [
        ("mode", 0o600),
        ("mtime_ns", 1_700_000_002_000_000_000),
    ],
)
def test_identical_bytes_with_different_archive_metadata_are_not_exact(
    metadata_field: str,
    different_value: int,
) -> None:
    source = {"files": [_manifest_item()]}
    destination = {"files": [_manifest_item(**{metadata_field: different_value})]}

    diff = copy_tools.diff_manifests(source, destination, sha256=True)

    assert diff.exact == ()
    assert diff.mismatched == ("payload.bin",)
    assert diff.transfer_paths == ("payload.bin",)


@pytest.mark.parametrize("transfer_required", [False, True], ids=["no-op", "after-transfer"])
def test_sha256_verified_copy_rehashes_source_at_completion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    transfer_required: bool,
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    source_file = _write_file(src, "payload.bin")
    dst.mkdir()
    if not transfer_required:
        shutil.copy2(source_file, dst / "payload.bin")

    source_hash_reads: list[bool] = []
    original_read_manifest = copy_tools.read_manifest

    def recording_read_manifest(
        endpoint: copy_tools.Endpoint,
        *,
        sha256: bool,
        runner: Any,
    ) -> dict[str, Any]:
        if endpoint.path == str(src):
            source_hash_reads.append(sha256)
        return original_read_manifest(endpoint, sha256=sha256, runner=runner)

    monkeypatch.setattr(copy_tools, "read_manifest", recording_read_manifest)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert transfer_required
        assert argv[0] == "rsync"
        assert kwargs["input"] == "payload.bin\0"
        shutil.copy2(source_file, dst / "payload.bin")
        return _ok()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--json"],
        runner=runner,
    )

    assert rc == 0
    assert source_hash_reads == [True, True]
    payload = _single_json_object(capsys.readouterr())
    assert payload["verify"]["source_stable"] is True


def test_source_partial_directory_name_is_not_hidden(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    relative = f"{copy_tools.PARTIAL_DIR_NAME}/payload.bin"
    _write_file(src, relative)
    dst.mkdir()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--dry-run", "--json"],
        runner=_unexpected_runner,
    )

    assert rc == 0
    payload = _single_json_object(capsys.readouterr())
    assert payload["plan"]["source_files"] == 1
    assert payload["plan"]["transfer_paths"] == [relative]


def test_verified_copy_explicitly_rejects_empty_source_directories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_file(src, "payload.bin")
    (src / "empty").mkdir()
    dst.mkdir()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--json"],
        runner=_unexpected_runner,
    )

    assert rc == 2
    payload = _single_json_object(capsys.readouterr())
    assert payload["ok"] is False
    assert "does not support empty source directories" in payload["error"]
    assert "empty" in payload["error"]


@pytest.mark.parametrize("failure", ["validation", "rsync"])
def test_copy_json_failures_emit_one_object_without_plain_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    dst = tmp_path / "dst"
    dst.mkdir()
    if failure == "validation":
        argv = [
            "copy",
            str(tmp_path / "missing"),
            str(dst),
            "--verify",
            "sha256",
            "--json",
        ]
        runner = _unexpected_runner
        expected_returncode = 2
    else:
        src = tmp_path / "src"
        _write_file(src, "payload.bin")
        argv = ["copy", str(src), str(dst), "--json"]

        def runner(command: list[str], **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(returncode=23, stdout="", stderr="rsync failed\n")

        expected_returncode = 23

    rc = main_cli.main(argv, runner=runner)

    assert rc == 2
    payload = _single_json_object(capsys.readouterr())
    assert payload["ok"] is False
    assert payload["returncode"] == expected_returncode
    assert {"mode", "ok", "message"} <= payload["verify"].keys()


def test_verified_and_unverified_dry_run_json_share_core_status_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_file(src, "payload.bin")
    dst.mkdir()

    assert main_cli.main(
        ["copy", str(src), str(dst), "--dry-run", "--json"],
        runner=_unexpected_runner,
    ) == 0
    unverified = _single_json_object(capsys.readouterr())

    assert main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--dry-run", "--json"],
        runner=_unexpected_runner,
    ) == 0
    verified = _single_json_object(capsys.readouterr())

    core_fields = {"ok", "dry_run", "returncode", "stdout", "stderr", "plan", "attempts", "verify"}
    assert core_fields <= unverified.keys()
    assert core_fields <= verified.keys()
    assert {field: unverified[field] for field in core_fields - {"plan", "verify"}} == {
        field: verified[field] for field in core_fields - {"plan", "verify"}
    }
    assert set(unverified["verify"]) == set(verified["verify"]) == {"mode", "ok", "message"}
    assert unverified["verify"]["mode"] == "none"
    assert verified["verify"]["mode"] == "sha256"
    assert unverified["verify"]["ok"] is None
    assert verified["verify"]["ok"] is None
