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


def _manifest(
    *,
    uid: int = 1000,
    gid: int = 1000,
    unsupported: tuple[dict[str, str], ...] = (),
    empty_directories: tuple[str, ...] = (),
) -> dict[str, Any]:
    data = b"payload"
    return {
        "schema_version": 1,
        "exists": True,
        "root_kind": "directory",
        "file_count": 1,
        "total_bytes": len(data),
        "files": [
            {
                "path": "payload.bin",
                "bytes": len(data),
                "kind": "file",
                "mode": 0o640,
                "mtime_ns": 1_700_000_000_000_000_000,
                "uid": uid,
                "gid": gid,
                "inode": 12345,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
        "unsupported": list(unsupported),
        "empty_directories": list(empty_directories),
    }


def _write_file(root: Path, relative: str, data: bytes = b"payload") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _result(returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


@pytest.mark.parametrize(("field", "value"), [("uid", 2000), ("gid", 2000)])
def test_verified_equality_ignores_ownership_but_hardlink_reuse_requires_it(
    field: str,
    value: int,
) -> None:
    source = _manifest()
    different_owner = _manifest(**{field: value})

    diff = copy_tools.diff_manifests(source, different_owner, sha256=True)

    assert diff.exact == ("payload.bin",)
    assert diff.transfer_paths == ()
    assert copy_tools.hardlinkable_paths(source, different_owner, diff.exact) == ()
    assert copy_tools.hardlinkable_paths(source, source, diff.exact) == ("payload.bin",)


@pytest.mark.parametrize(
    ("change", "after"),
    [
        ("unsupported entry", _manifest(unsupported=({"path": "late-link", "kind": "symlink"},))),
        ("empty directory", _manifest(empty_directories=("late-empty",))),
    ],
)
def test_source_snapshot_stability_rejects_late_non_file_entries(
    change: str,
    after: dict[str, Any],
) -> None:
    before = _manifest()

    assert copy_tools.source_snapshot_stable(before, before, sha256=True)
    assert not copy_tools.source_snapshot_stable(before, after, sha256=True), change


def test_partial_filter_only_removes_destination_extras() -> None:
    partial_source_path = f"{copy_tools.PARTIAL_DIR_NAME}/source-owned.bin"
    partial_destination_extra = f"{copy_tools.PARTIAL_DIR_NAME}/interrupted.bin"
    diff = copy_tools.ManifestDiff(
        exact=("exact.bin",),
        missing=(partial_source_path,),
        mismatched=("changed.bin",),
        extra=(partial_destination_extra, "real-extra.bin"),
    )

    filtered = copy_tools.ignore_destination_internal_partials(diff, enabled=True)

    assert filtered.exact == diff.exact
    assert filtered.missing == (partial_source_path,)
    assert filtered.mismatched == diff.mismatched
    assert filtered.extra == ("real-extra.bin",)


def test_partial_state_from_failed_invocation_allows_later_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    source_file = _write_file(src, "payload.bin", b"complete payload")
    dst.mkdir()
    partial_file = dst / copy_tools.PARTIAL_DIR_NAME / "payload.bin"
    invocations: list[str] = []

    def interrupted(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        invocations.append("interrupted")
        assert argv[0] == "rsync"
        assert f"--partial-dir={copy_tools.PARTIAL_DIR_NAME}" in argv
        assert kwargs["input"] == "payload.bin\0"
        _write_file(dst, f"{copy_tools.PARTIAL_DIR_NAME}/payload.bin", b"complete")
        return _result(23, "interrupted\n")

    command = [
        "copy",
        str(src),
        str(dst),
        "--verify",
        "sha256",
        "--partial",
        "--retries",
        "0",
        "--json",
    ]
    assert main_cli.main(command, runner=interrupted) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["attempts"][0]["remaining"] == ["payload.bin"]
    assert partial_file.read_bytes() == b"complete"

    def resumed(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        invocations.append("resumed")
        assert argv[0] == "rsync"
        assert f"--partial-dir={copy_tools.PARTIAL_DIR_NAME}" in argv
        assert kwargs["input"] == "payload.bin\0"
        assert partial_file.read_bytes() == b"complete"
        shutil.copy2(source_file, dst / "payload.bin")
        return _result()

    assert main_cli.main(command, runner=resumed) == 0
    second = json.loads(capsys.readouterr().out)

    assert invocations == ["interrupted", "resumed"]
    assert second["plan"]["destination_extra"] == []
    assert second["plan"]["transfer_paths"] == ["payload.bin"]
    assert second["attempts"][0]["paths"] == ["payload.bin"]
    assert second["verify"]["ok"] is True
