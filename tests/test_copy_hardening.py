from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import main_cli


def _ok() -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _unexpected_transfer(argv: list[str], **kwargs: Any) -> SimpleNamespace:
    raise AssertionError(f"copy must fail before transfer: {argv}")


def _write_file(root: Path, relative: str, data: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("location", "kind"),
    [
        ("source", "symlink"),
        ("source", "special"),
        ("destination", "symlink"),
        ("destination", "special"),
    ],
)
def test_verified_copy_rejects_symlinks_and_special_files_before_transfer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    location: str,
    kind: str,
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_file(src, "payload.bin", b"payload")
    dst.mkdir()
    root = src if location == "source" else dst
    unsupported = root / "unsupported"
    if kind == "symlink":
        unsupported.symlink_to("missing-target")
    else:
        os.mkfifo(unsupported)
    destination_entries = sorted(path.name for path in dst.iterdir())

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--json"],
        runner=_unexpected_transfer,
    )

    assert rc == 2
    assert sorted(path.name for path in dst.iterdir()) == destination_entries
    captured = capsys.readouterr()
    assert captured.err == ""
    error = json.loads(captured.out)["error"]
    assert f"does not support {location} symlinks or special files" in error
    assert f"unsupported ({kind})" in error


@pytest.mark.parametrize("direction", ["destination-inside-source", "source-inside-destination"])
def test_verified_copy_rejects_overlapping_directory_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    direction: str,
) -> None:
    if direction == "destination-inside-source":
        src = tmp_path / "tree"
        dst = src / "destination"
    else:
        dst = tmp_path / "tree"
        src = dst / "source"
    _write_file(src, "payload.bin", b"payload")
    dst.mkdir(parents=True, exist_ok=True)

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--json"],
        runner=_unexpected_transfer,
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "requires non-overlapping source and destination roots" in json.loads(captured.out)["error"]


def test_verified_copy_rejects_destination_extras_before_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_file(src, "payload.bin", b"payload")
    extra = _write_file(dst, "keep.bin", b"keep-me")

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--json"],
        runner=_unexpected_transfer,
    )

    assert rc == 2
    assert extra.read_bytes() == b"keep-me"
    assert not (dst / "payload.bin").exists()
    captured = capsys.readouterr()
    assert captured.err == ""
    error = json.loads(captured.out)["error"]
    assert "destination contains files absent from the source" in error
    assert "keep.bin" in error


def test_parse_endpoint_preserves_local_trailing_slash(tmp_path: Path) -> None:
    raw = f"{tmp_path / 'source'}/"

    endpoint = copy_tools.parse_endpoint(raw)

    assert endpoint.host is None
    assert endpoint.raw == raw
    assert endpoint.path == raw
    assert endpoint.rsync_spec() == raw


def test_selective_directory_rsync_uses_hardened_partial_and_comparison_flags(tmp_path: Path) -> None:
    src = copy_tools.parse_endpoint(str(tmp_path / "src"))
    dst = copy_tools.parse_endpoint(str(tmp_path / "dst"))

    argv = copy_tools.build_selective_rsync_argv(
        src,
        dst,
        source_is_directory=True,
        partial=True,
    )

    assert "--ignore-times" in argv
    assert "--partial-dir=.ucl-rsync-partial" in argv
    assert "--partial" not in argv
    assert argv[-2:] == [f"{src.path}/", f"{dst.path}/"]


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_real_selective_rsync_replaces_same_size_same_mtime_content(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    source_file = _write_file(src, "payload.bin", b"source")
    destination_file = _write_file(dst, "payload.bin", b"DESTIN")
    timestamp_ns = 1_700_000_000_000_000_000
    os.utime(source_file, ns=(timestamp_ns, timestamp_ns))
    os.utime(destination_file, ns=(timestamp_ns, timestamp_ns))
    assert source_file.stat().st_size == destination_file.stat().st_size
    assert source_file.stat().st_mtime_ns == destination_file.stat().st_mtime_ns

    argv = copy_tools.build_selective_rsync_argv(
        copy_tools.parse_endpoint(str(src)),
        copy_tools.parse_endpoint(str(dst)),
        source_is_directory=True,
    )
    rsync = shutil.which("rsync")
    assert rsync is not None
    argv[0] = rsync

    proc = subprocess.run(
        argv,
        input=copy_tools.files_from_input(["payload.bin"]),
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert destination_file.read_bytes() == b"source"


@pytest.mark.parametrize("metadata_difference", ["mode", "mtime"])
def test_verified_copy_refuses_hardlink_reuse_when_metadata_differs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    metadata_difference: str,
) -> None:
    src = tmp_path / "src"
    reuse = tmp_path / "reuse"
    dst = tmp_path / "dst"
    source_file = _write_file(src, "payload.bin", b"payload")
    reuse_file = _write_file(reuse, "payload.bin", b"payload")
    dst.mkdir()
    timestamp_ns = 1_700_000_000_000_000_000
    source_file.chmod(0o640)
    reuse_file.chmod(0o640)
    os.utime(source_file, ns=(timestamp_ns, timestamp_ns))
    os.utime(reuse_file, ns=(timestamp_ns, timestamp_ns))
    if metadata_difference == "mode":
        reuse_file.chmod(0o600)
    else:
        os.utime(reuse_file, ns=(timestamp_ns + 2_000_000_000,) * 2)

    def transfer(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv[0] == "rsync"
        assert kwargs["input"] == "payload.bin\0"
        shutil.copy2(source_file, dst / "payload.bin")
        return _ok()

    rc = main_cli.main(
        [
            "copy",
            str(src),
            str(dst),
            "--verify",
            "sha256",
            "--reuse-from",
            str(reuse),
            "--json",
        ],
        runner=transfer,
    )

    assert rc == 0
    destination_file = dst / "payload.bin"
    assert destination_file.stat().st_ino != reuse_file.stat().st_ino
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["reuse_candidates"] == []
    assert payload["plan"]["reused"] == []
    assert payload["plan"]["transfer_paths"] == ["payload.bin"]


def test_verified_copy_detects_source_mutation_during_transfer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    source_file = _write_file(src, "payload.bin", b"before")
    dst.mkdir()
    timestamp_ns = 1_700_000_000_000_000_000
    os.utime(source_file, ns=(timestamp_ns, timestamp_ns))

    def transfer_then_mutate(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv[0] == "rsync"
        assert kwargs["input"] == "payload.bin\0"
        shutil.copy2(source_file, dst / "payload.bin")
        source_file.write_bytes(b"AFTER!")
        os.utime(source_file, ns=(timestamp_ns + 2_000_000_000,) * 2)
        return _ok()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--retries", "0", "--json"],
        runner=transfer_then_mutate,
    )

    assert rc == 2
    assert (dst / "payload.bin").read_bytes() == b"before"
    payload = json.loads(capsys.readouterr().out)
    assert payload["verify"]["ok"] is False
    assert payload["verify"]["source_stable"] is False
    assert payload["verify"]["transfer_paths"] == []
    assert payload["verify"]["message"] == "source changed during copy; result is not trusted"
