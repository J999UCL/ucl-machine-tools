from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import main_cli
from ucl_machine_tools import ssh as ssh_tools


def ok(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def failed(returncode: int = 23, stderr: str = "rsync partial transfer\n") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(files: dict[str, bytes], *, exists: bool = True) -> dict[str, Any]:
    rows = [
        {"path": path, "bytes": len(data), "sha256": sha256(data)}
        for path, data in sorted(files.items())
    ]
    return {
        "schema_version": 1,
        "exists": exists,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def manifest_stdout(payload: dict[str, Any]) -> str:
    return "\n".join([copy_tools.MANIFEST_BEGIN, json.dumps(payload), copy_tools.MANIFEST_END])


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def selected_files(argv: list[str], kwargs: dict[str, Any]) -> list[str]:
    if argv[0] == "rsync":
        rsync_argv = argv
    elif argv[0] == "python3":
        rsync_argv = argv[argv.index("rsync") :]
    else:
        assert argv[:6] == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
        assert argv[7:9] == ["bash", "-lc"]
        rsync_argv = shlex.split(shlex.split(argv[9])[0])
    assert "--files-from=-" in rsync_argv
    assert "--from0" in rsync_argv
    raw = kwargs.get("input", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return [path for path in raw.split("\0") if path]


def copy_selected(src: Path, dst: Path, paths: list[str]) -> None:
    for relative in paths:
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / relative, target)


def write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "gpu_class": "3090ti"},
                    "barnacle-l": {"ssh": "barnacle.internal", "gpu_class": "3090ti"},
                }
            }
        ),
        encoding="utf-8",
    )
    return catalog


def test_diff_manifests_uses_sha256_to_classify_exact_missing_and_mismatched() -> None:
    source = manifest(
        {
            "changed.bin": b"source",
            "missing.bin": b"missing",
            "same.bin": b"same",
        }
    )
    destination = manifest(
        {
            "changed.bin": b"DESTIN",
            "same.bin": b"same",
        }
    )

    plan = copy_tools.diff_manifests(source, destination, sha256=True)

    assert plan.as_dict() == {
        "exact": ["same.bin"],
        "missing": ["missing.bin"],
        "mismatched": ["changed.bin"],
        "extra": [],
        "transfer_paths": ["changed.bin", "missing.bin"],
    }


def test_ucl_copy_reconcile_skips_an_already_exact_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    files = {"nested/a.bin": b"alpha", "nested/b.bin": b"beta"}
    write_tree(src, files)
    write_tree(dst, files)
    for relative in files:
        shutil.copystat(src / relative, dst / relative)

    def no_transfer(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise AssertionError(f"exact files must not be transferred: {argv}")

    rc = main_cli.main(["copy", str(src), str(dst), "--verify", "sha256", "--json"], runner=no_transfer)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["destination_exact"] == ["nested/a.bin", "nested/b.bin"]
    assert payload["plan"]["transfer_paths"] == []
    assert payload["attempts"] == []
    assert payload["verify"]["mode"] == "sha256"
    assert payload["verify"]["ok"] is True
    assert payload["verify"]["message"] == "ok"


def test_ucl_copy_reconcile_transfers_only_missing_and_mismatched_then_verifies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    write_tree(
        src,
        {
            "changed.bin": b"source",
            "nested/missing.bin": b"missing",
            "same.bin": b"same",
        },
    )
    write_tree(dst, {"changed.bin": b"DESTIN", "same.bin": b"same"})
    shutil.copystat(src / "changed.bin", dst / "changed.bin")
    shutil.copystat(src / "same.bin", dst / "same.bin")
    transferred: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        paths = selected_files(argv, kwargs)
        transferred.append(paths)
        copy_selected(src, dst, paths)
        return ok(stdout="copied\n")

    rc = main_cli.main(["copy", str(src), str(dst), "--verify", "sha256", "--json"], runner=runner)

    assert rc == 0
    assert transferred == [["changed.bin", "nested/missing.bin"]]
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["destination_exact"] == ["same.bin"]
    assert payload["plan"]["destination_missing"] == ["nested/missing.bin"]
    assert payload["plan"]["destination_mismatched"] == ["changed.bin"]
    assert payload["plan"]["transfer_paths"] == ["changed.bin", "nested/missing.bin"]
    assert payload["verify"]["transfer_paths"] == []
    assert payload["verify"]["ok"] is True


def test_ucl_copy_reconcile_can_hardlink_exact_reuse_files_into_clean_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    reuse = tmp_path / "reuse"
    dst = tmp_path / "dst"
    write_tree(src, {"changed.bin": b"source", "missing.bin": b"new", "same.bin": b"same"})
    write_tree(reuse, {"changed.bin": b"DESTIN", "same.bin": b"same"})
    shutil.copystat(src / "same.bin", reuse / "same.bin")
    dst.mkdir()
    transferred: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        paths = selected_files(argv, kwargs)
        transferred.append(paths)
        copy_selected(src, dst, paths)
        return ok()

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
        runner=runner,
    )

    assert rc == 0
    assert transferred == [["changed.bin", "missing.bin"]]
    assert (dst / "same.bin").stat().st_ino == (reuse / "same.bin").stat().st_ino
    assert (dst / "changed.bin").stat().st_ino != (reuse / "changed.bin").stat().st_ino
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["reuse_candidates"] == ["same.bin"]
    assert payload["plan"]["reused"] == ["same.bin"]
    assert payload["plan"]["transfer_paths"] == ["changed.bin", "missing.bin"]
    assert payload["verify"]["ok"] is True


def test_ucl_copy_reconcile_fails_when_final_sha256_verification_differs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    write_tree(src, {"a.bin": b"source"})
    dst.mkdir()

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert selected_files(argv, kwargs) == ["a.bin"]
        (dst / "a.bin").write_bytes(b"DESTIN")
        return ok()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--retries", "0", "--json"],
        runner=runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["verify"]["mode"] == "sha256"
    assert payload["verify"]["ok"] is False
    assert payload["verify"]["message"] == "sha256 manifest differs"
    assert payload["verify"]["transfer_paths"] == ["a.bin"]


def test_ucl_copy_reconcile_retries_only_files_still_failing_verification(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    write_tree(src, {"first.bin": b"first", "second.bin": b"second"})
    dst.mkdir()
    attempts: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        paths = selected_files(argv, kwargs)
        attempts.append(paths)
        if len(attempts) == 1:
            copy_selected(src, dst, ["first.bin"])
            return failed()
        copy_selected(src, dst, paths)
        return ok()

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--retries", "1", "--json"],
        runner=runner,
    )

    assert rc == 0
    assert attempts == [["first.bin", "second.bin"], ["second.bin"]]
    payload = json.loads(capsys.readouterr().out)
    assert [attempt["paths"] for attempt in payload["attempts"]] == attempts
    assert [attempt["returncode"] for attempt in payload["attempts"]] == [23, 0]
    assert payload["attempts"][0]["remaining"] == ["second.bin"]
    assert payload["attempts"][1]["remaining"] == []
    assert payload["verify"]["ok"] is True


def test_ucl_copy_reconcile_preserves_transfer_output_when_post_verify_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = manifest({"payload.bin": b"payload"})
    destination = manifest({})
    monkeypatch.setattr(
        main_cli,
        "_read_reconcile_manifests",
        lambda *args, **kwargs: {"source": source, "destination": destination},
    )
    monkeypatch.setattr(
        main_cli,
        "_copy_endpoint_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("manifest helper lost connection")),
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        del argv, kwargs
        return ok(
            stdout="VirtualBox is legitimate transfer stdout\n",
            stderr="VBoxManage is legitimate transfer stderr\n",
        )

    rc = main_cli.main(
        ["copy", "/tmp/source", "/tmp/destination", "--verify", "sha256", "--json"],
        runner=runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["stdout"] == "VirtualBox is legitimate transfer stdout\n"
    assert payload["stderr"] == "VBoxManage is legitimate transfer stderr\n"
    assert len(payload["attempts"]) == 1
    assert payload["attempts"][0]["verification_error"] == "manifest helper lost connection"
    assert payload["verify"]["message"] == "post-transfer verification failed"
    assert payload["error"] == "manifest helper lost connection"


def test_ucl_copy_human_error_reports_transfer_output_and_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = manifest({"payload.bin": b"payload"})
    destination = manifest({})
    monkeypatch.setattr(
        main_cli,
        "_read_reconcile_manifests",
        lambda *args, **kwargs: {"source": source, "destination": destination},
    )
    monkeypatch.setattr(
        main_cli,
        "_copy_endpoint_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("manifest helper lost connection")),
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        del argv, kwargs
        return ok(stdout="transfer stdout\n", stderr="transfer stderr\n")

    rc = main_cli.main(
        ["copy", "/tmp/source", "/tmp/destination", "--verify", "sha256"],
        runner=runner,
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "transfer stdout\n" in captured.out
    assert "verify: post-transfer verification failed\n" in captured.out
    assert "transfer stderr\n" in captured.err
    assert "error: manifest helper lost connection\n" in captured.err


def test_ucl_copy_reconcile_dry_run_reports_plan_without_mutating_or_transferring(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    write_tree(src, {"missing.bin": b"missing", "same.bin": b"same"})
    write_tree(dst, {"same.bin": b"same"})
    shutil.copystat(src / "same.bin", dst / "same.bin")

    def no_transfer(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise AssertionError(f"dry-run must not execute a transfer: {argv}")

    rc = main_cli.main(
        ["copy", str(src), str(dst), "--verify", "sha256", "--dry-run", "--json"],
        runner=no_transfer,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["plan"]["destination_exact"] == ["same.bin"]
    assert payload["plan"]["transfer_paths"] == ["missing.bin"]
    assert payload["attempts"] == []
    assert not (dst / "missing.bin").exists()


def test_ucl_copy_reconcile_remote_to_remote_manifests_and_transfer_stay_on_endpoint_hosts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = write_catalog(tmp_path)
    source = manifest({"missing.bin": b"missing", "same.bin": b"same"})
    destination_before = manifest({"same.bin": b"same"})
    destination_after = source
    destination_reads = 0
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent-test.sock")
    transfer_hosts: list[str] = []
    transfers: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        nonlocal destination_reads
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh-add", "-l"]:
            return ok(stdout="256 SHA256:test controller-key\n")
        if argv == copy_tools.build_remote_destination_probe_argv("barbury.internal", "barnacle.internal"):
            return ok()
        if argv == ssh_tools.build_remote_python_argv("barbury.internal"):
            assert "ROOT=\"/tmp/src\"" in kwargs["input"]
            assert "SHA256=True" in kwargs["input"]
            return ok(stdout=manifest_stdout(source))
        if argv == ssh_tools.build_remote_python_argv("barnacle.internal"):
            assert "ROOT=\"/tmp/dst\"" in kwargs["input"]
            assert "SHA256=True" in kwargs["input"]
            payload = destination_before if destination_reads == 0 else destination_after
            destination_reads += 1
            return ok(stdout=manifest_stdout(payload))
        assert argv[0] == "python3"
        assert "barbury.internal" in argv
        transfer_hosts.append("barbury.internal")
        transfers.append(selected_files(argv, kwargs))
        return ok()

    rc = main_cli.main(
        [
            "copy",
            "barbury-l:/tmp/src",
            "barnacle-l:/tmp/dst",
            "--catalog",
            str(catalog),
            "--verify",
            "sha256",
            "--json",
        ],
        runner=runner,
    )

    assert rc == 0
    assert destination_reads == 2
    assert transfer_hosts == ["barbury.internal"]
    assert transfers == [["missing.bin"]]
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "remote-to-remote"
    assert payload["plan"]["destination_exact"] == ["same.bin"]
    assert payload["plan"]["transfer_paths"] == ["missing.bin"]
    assert payload["verify"]["ok"] is True
