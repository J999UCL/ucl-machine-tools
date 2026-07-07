from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import launch, main_cli
from ucl_machine_tools.registry import RunRecord, write_record


def ok(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def tmux_stdout(sessions: list[str]) -> str:
    return "\n".join(
        [
            launch.TMUX_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "sessions": sessions}),
            launch.TMUX_SENTINEL_END,
        ]
    )


def inventory_stdout(host: str = "barbury-l", *, busy: bool = False) -> str:
    gpu = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "memory_total_mb": 24576,
        "memory_used_mb": 1024,
        "memory_free_mb": 23552,
        "utilization_gpu_percent": 1,
        "processes": [{"pid": 7}] if busy else [],
    }
    payload = {
        "schema_version": 1,
        "host": host,
        "hostname": host,
        "ok": True,
        "gpus": [gpu],
        "filesystems": [{"path": "/tmp", "available_gb": 500}],
        "scratch": {"root": "/tmp/ucl-machine-tools", "exists": True},
        "restart": {"policy": "lab_pc", "text": "Mon/Thu 19:30-midnight; may reboot anytime"},
        "errors": [],
    }
    return "\n".join(
        [
            "login noise",
            "UCL_INVENTORY_JSON_BEGIN",
            json.dumps(payload),
            "UCL_INVENTORY_JSON_END",
        ]
    )


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    return bundle


class FakeStdout:
    closed = False

    def close(self) -> None:
        self.closed = True


class FakePopen:
    calls: list[list[str]] = []

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.__class__.calls.append(argv)
        self.stdout = FakeStdout()

    def wait(self) -> int:
        return 0


def test_ucl_status_routes_inventory_json(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        return ok(stdout=inventory_stdout())

    assert main_cli.main(["status", "barbury-l", "--json"], runner=runner) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["hosts"][0]["host"] == "barbury-l"
    assert calls[0] == ["ssh", "-O", "check", "knuckles"]


def test_ucl_run_full_fake_path_writes_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    bundle = make_bundle(tmp_path)
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "barbury-l", "python3", "-"] and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            assert "exec > >(tee -a" in kwargs["input"]
            return ok()
        if "tmux new-window" in joined:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        ["run", "--host", "barbury-l", "--local-dir", str(bundle), "--script", "run.sh", "--arg", "x"],
        runner=runner,
        popener=FakePopen,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "session:    work" in out
    latest = tmp_path / "cache" / "runs" / "latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8"))["kind"] == "run"


def test_ucl_exec_defaults_to_single_existing_tmux_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "barbury-l", "python3", "-"] and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "mkdir -p" in joined and "tar -xf" not in joined:
            return ok()
        if "cat >" in joined:
            assert "hostname" in kwargs["input"]
            return ok()
        if "tmux new-window" in joined:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--", "hostname"], runner=runner) == 0

    assert "session:    work" in capsys.readouterr().out
    assert any("tmux new-window" in " ".join(call) for call in calls)


def test_ucl_exec_requires_explicit_session_when_no_existing_tmux(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "barbury-l", "python3", "-"] and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout([]))
        raise AssertionError(f"unexpected argv after failed tmux decision: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--", "hostname"], runner=runner) == 2

    assert "no tmux sessions exist" in capsys.readouterr().err
    assert not any("cat >" in " ".join(call) for call in calls)


def test_ucl_exec_stdin_dry_run_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeStdin:
        def read(self) -> str:
            return "echo hello\n"

    monkeypatch.setattr("sys.stdin", FakeStdin())

    rc = main_cli.main(["exec", "barbury-l", "--stdin", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "dry_run: true" in out
    assert "shell:      bash" in out


def test_ucl_doctor_reports_host_state(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "barbury-l", "python3", "-"] and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if argv == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=8", "barbury-l", "python3", "-"]:
            return ok(stdout=inventory_stdout())
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["doctor", "barbury-l"], runner=runner) == 0

    out = capsys.readouterr().out
    assert "status:        ready" in out
    assert "tmux_sessions: work" in out


def test_profile_flags_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_cli.main(["exec", "barbury-l", "--profile", "uv", "--dry-run", "--", "hostname"]) == 2
    assert "--profile" in capsys.readouterr().err


def test_ucl_tail_filters_login_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    write_record(
        RunRecord(
            run_id="demo",
            kind="exec",
            host="barbury-l",
            ssh_host="barbury-l",
            session="demo",
            window="exec_demo",
            remote_dir="/tmp/ucl-machine-tools/launchers/demo",
            log_path="/tmp/ucl-machine-tools/launchers/demo/run.log",
            command=("hostname",),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "barbury-l", "python3", "-"]:
            assert "/tmp/ucl-machine-tools/launchers/demo/run.log" in kwargs["input"]
            return ok(
                stdout="\n".join(
                    [
                        "VBoxManage startup noise",
                        main_cli.TAIL_SENTINEL_BEGIN,
                        "actual log line",
                        main_cli.TAIL_SENTINEL_END,
                        "logout noise",
                    ]
                )
                + "\n"
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--lines", "5"], runner=runner) == 0

    assert capsys.readouterr().out == "actual log line\n"


def test_help_exposes_unified_commands_and_not_legacy_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "status" in help_text
    assert "exec" in help_text
    assert "ucl-inventory" not in help_text
    assert "ucl-launch" not in help_text
