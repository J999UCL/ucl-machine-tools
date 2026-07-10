from __future__ import annotations

import ast
import base64
import io
import json
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import envcheck, launch, main_cli
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


def remote_python_argv(host: str = "barbury-l", *, timeout_seconds: int | None = None) -> list[str]:
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
    if timeout_seconds is not None:
        argv += ["-o", f"ConnectTimeout={timeout_seconds}"]
    return [*argv, host, "python3", "-"]


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


def write_status_catalog(tmp_path: Path) -> Path:
    path = tmp_path / "ucl_hosts.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {"scratch_root": "/tmp/ucl-machine-tools"},
                "groups": {
                    "3090ti": ["barbury-l", "canada-l"],
                    "timeshare": ["cream"],
                },
                "hosts": {
                    "barbury-l": {"gpu_class": "3090ti", "restart_policy": "lab_pc"},
                    "canada-l": {"gpu_class": "3090ti", "restart_policy": "lab_pc"},
                    "cream": {"gpu_class": "rtx6000", "restart_policy": "timeshare"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def exec_stdout(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    timed_out: bool = False,
    wrapper_error: bool = False,
    outside_noise: str = "VBoxManage: error: wrapper noise\n",
) -> str:
    payload = {
        "schema_version": 1,
        "returncode": returncode,
        "stdout_b64": base64.b64encode(stdout).decode("ascii"),
        "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        "timed_out": timed_out,
        "wrapper_error": wrapper_error,
    }
    return "\n".join(
        [
            outside_noise.rstrip("\n"),
            main_cli.EXEC_SENTINEL_BEGIN,
            json.dumps(payload),
            main_cli.EXEC_SENTINEL_END,
            "logout noise",
        ]
    )


def embedded_exec_params(source: str) -> dict[str, Any]:
    match = re.search(r"PARAMS=json\.loads\((?P<literal>.*)\)", source)
    assert match is not None
    return json.loads(ast.literal_eval(match.group("literal")))


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


def test_ucl_status_accepts_multiple_positional_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)
    probed: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        host = argv[-3]
        probed.append(host)
        return ok(stdout=inventory_stdout(host=host))

    rc = main_cli.main(["status", "barbury-l", "canada-l", "--catalog", str(catalog), "--json"], runner=runner)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["hosts"]] == ["barbury-l", "canada-l"]
    assert probed == ["barbury-l", "canada-l"]


def test_ucl_status_modes_accept_multiple_targets_and_selector_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)
    calls: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        host = argv[-3]
        calls.append(host)
        return ok(stdout=inventory_stdout(host=host, busy=(host == "canada-l")))

    assert (
        main_cli.main(
            ["status", "recommend", "barbury-l", "canada-l", "--catalog", str(catalog), "--json"],
            runner=runner,
        )
        == 0
    )
    recommend_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in recommend_payload["hosts"]] == ["barbury-l"]
    assert calls == ["barbury-l", "canada-l"]

    calls.clear()
    assert (
        main_cli.main(
            ["status", "gpus", "barbury-l", "canada-l", "--catalog", str(catalog), "--json"],
            runner=runner,
        )
        == 0
    )
    gpus_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in gpus_payload["hosts"]] == ["barbury-l", "canada-l"]
    assert calls == ["barbury-l", "canada-l"]

    calls.clear()
    assert (
        main_cli.main(
            ["status", "--selector", "barbury-l", "canada-l", "--catalog", str(catalog), "--json"],
            runner=runner,
        )
        == 0
    )
    override_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in override_payload["hosts"]] == ["barbury-l"]
    assert calls == ["barbury-l"]


def test_ucl_status_expands_multiple_selectors_in_catalog_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)
    probed: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        host = argv[-3]
        probed.append(host)
        return ok(stdout=inventory_stdout(host=host))

    assert main_cli.main(["status", "3090ti", "timeshare", "--catalog", str(catalog), "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["hosts"]] == ["barbury-l", "canada-l", "cream"]
    assert probed == ["barbury-l", "canada-l", "cream"]

    probed.clear()
    assert main_cli.main(["status", "barbury-l", "3090ti", "--catalog", str(catalog), "--json"], runner=runner) == 0
    dedupe_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in dedupe_payload["hosts"]] == ["barbury-l", "canada-l"]
    assert probed == ["barbury-l", "canada-l"]

    probed.clear()
    assert main_cli.main(["status", "all", "!cream", "--catalog", str(catalog), "--json"], runner=runner) == 0
    exclusion_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in exclusion_payload["hosts"]] == ["barbury-l", "canada-l"]
    assert probed == ["barbury-l", "canada-l"]


def test_ucl_status_defaults_to_all_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)
    probed: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        host = argv[-3]
        probed.append(host)
        return ok(stdout=inventory_stdout(host=host))

    assert main_cli.main(["status", "--catalog", str(catalog), "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["hosts"]] == ["barbury-l", "canada-l", "cream"]
    assert probed == ["barbury-l", "canada-l", "cream"]


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
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
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


def test_ucl_exec_sync_command_prints_output_and_preserves_argv(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            source = kwargs["input"]
            assert embedded_exec_params(source)["argv"] == ["python3", "-c", 'print("hi")', "--remote-flag"]
            assert "shell=True" not in source
            return ok(stdout=exec_stdout(stdout=b"hi\n", stderr=b"VirtualBox from command stderr\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "python3", "-c", 'print("hi")', "--remote-flag"], runner=runner) == 0

    captured = capsys.readouterr()
    assert captured.out == "hi\n"
    assert captured.err == "VirtualBox from command stderr\n"
    assert not any("UCL_TMUX_JSON_BEGIN" in str(call) for call in calls)


def test_ucl_exec_sync_accepts_multiple_hosts_with_delimiter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv[-2:] == ["python3", "-"]:
            assert "-o" in argv
            assert "ConnectTimeout=30" in argv
            host = argv[-3]
            params = embedded_exec_params(kwargs["input"])
            assert params["argv"] == ["hostname"]
            assert params["cwd"] == "/tmp"
            assert params["timeout"] == 2.0
            assert params["env"]["DEMO"] == "1"
            return ok(stdout=exec_stdout(stdout=f"{host}\n".encode()))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        [
            "exec",
            "barbury-l",
            "canada-l",
            "--catalog",
            str(catalog),
            "--cwd",
            "/tmp",
            "--timeout",
            "2",
            "--env",
            "DEMO=1",
            "--json",
            "--",
            "hostname",
        ],
        runner=runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["results"]] == ["barbury-l", "canada-l"]
    assert [row["stdout"] for row in payload["results"]] == ["barbury-l\n", "canada-l\n"]


def test_ucl_exec_single_host_command_can_contain_delimiter(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            params = embedded_exec_params(kwargs["input"])
            assert params["argv"] == ["python3", "--", "-c"]
            return ok(stdout=exec_stdout(stdout=b"ok\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(["exec", "barbury-l", "python3", "--", "-c"], runner=runner)

    assert rc == 0
    assert capsys.readouterr().out == "ok\n"


def test_ucl_exec_sync_expands_selector_to_multiple_hosts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv[-2:] == ["python3", "-"]:
            host = argv[-3]
            return ok(stdout=exec_stdout(stdout=f"{host}\n".encode()))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(["exec", "3090ti", "--catalog", str(catalog), "--json", "--", "hostname"], runner=runner)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["results"]] == ["barbury-l", "canada-l"]


def test_ucl_exec_rejects_detach_with_multiple_hosts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)

    rc = main_cli.main(["exec", "barbury-l", "canada-l", "--catalog", str(catalog), "--detach", "--", "hostname"])

    assert rc == 2
    assert "multi-host exec is synchronous only" in capsys.readouterr().err


def test_ucl_exec_sync_supports_options_cwd_json_timeout_and_gpu_auto(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=8", "barbury-l", "python3", "-"]:
            return ok(stdout=inventory_stdout())
        if argv == remote_python_argv(timeout_seconds=30):
            source = kwargs["input"]
            params = embedded_exec_params(source)
            assert params["cwd"] == "/tmp"
            assert params["timeout"] == 1.0
            assert params["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            return ok(stdout=exec_stdout(stdout=b"/tmp\n", returncode=0))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        ["exec", "barbury-l", "--gpu", "auto", "--cwd", "/tmp", "--timeout", "1", "--json", "pwd"],
        runner=runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stdout"] == "/tmp\n"
    assert payload["stderr"] == ""
    assert payload["returncode"] == 0
    assert payload["timed_out"] is False


def test_ucl_exec_sync_separates_command_and_connect_timeouts(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=7):
            params = embedded_exec_params(kwargs["input"])
            assert params["timeout"] == 9.0
            return ok(stdout=exec_stdout(stdout=b"ok\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--timeout", "9", "--connect-timeout", "7", "hostname"], runner=runner) == 0
    assert capsys.readouterr().out == "ok\n"


def test_ucl_exec_sync_accepts_delimiter_for_dash_command_and_timeout_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv():
            params = embedded_exec_params(kwargs["input"])
            assert params["argv"] == ["-remote-command", "arg"]
            assert params["timeout"] == 0.0
            return ok(stdout=exec_stdout(stdout=b"dash-ok\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--timeout", "0", "--connect-timeout", "0", "--", "-remote-command", "arg"], runner=runner) == 0
    assert capsys.readouterr().out == "dash-ok\n"


def test_ucl_exec_sync_returns_remote_nonzero_and_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    responses = [
        exec_stdout(stderr=b"bad\n", returncode=7),
        exec_stdout(stderr=b"ucl exec timed out after 1.0 seconds\n", returncode=124, timed_out=True),
    ]

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(stdout=responses.pop(0))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "false"], runner=runner) == 7
    assert capsys.readouterr().err == "bad\n"
    assert main_cli.main(["exec", "barbury-l", "--timeout", "1", "sleep", "5"], runner=runner) == 124
    assert "timed out" in capsys.readouterr().err


def test_ucl_exec_sync_reports_empty_ssh_255_smartly(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(returncode=255, stdout="", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "SSH failed before remote exec wrapper started on barbury-l (exit 255)" in err
    assert "no stderr/stdout" in err


def test_ucl_exec_sync_reports_refused_jump_forwarding(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr=(
                    "Stdio forwarding request failed: Session open refused by peer\n"
                    "Connection closed by UNKNOWN port 65535\n"
                ),
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "ProxyJump/control-master forwarding was refused" in err
    assert "knuckles control master may be stale" in err


def test_ucl_exec_sync_reports_no_route_to_host(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="channel 0: open failed: connect failed: No route to host\nstdio forwarding failed\n",
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "target host is unreachable from the jump host" in err
    assert "No route to host" in err


def test_ucl_exec_sync_reports_wrapper_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(returncode=127, stdout="", stderr="python3: command not found\n")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "Remote exec wrapper failed before it could return a result on barbury-l (exit 127)" in err
    assert "python3: command not found" in err


def test_ucl_exec_sync_reports_missing_sentinel(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(stdout="not sentinel output\n", stderr="wrapper warning\n")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "Remote exec wrapper on barbury-l did not return sentinel JSON" in err
    assert "wrapper warning" in err
    assert "not sentinel output" in err


def test_ucl_exec_sync_reports_malformed_sentinel_json(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(
                stdout="\n".join(
                    [
                        main_cli.EXEC_SENTINEL_BEGIN,
                        "{not-json",
                        main_cli.EXEC_SENTINEL_END,
                    ]
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    assert "malformed sentinel JSON" in capsys.readouterr().err


def test_ucl_exec_sync_distinguishes_wrapper_error_payload(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(
                stdout=exec_stdout(
                    stderr=b"FileNotFoundError: [Errno 2] No such file or directory: 'missing-command'\n",
                    returncode=127,
                    wrapper_error=True,
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "missing-command"], runner=runner) == 127

    err = capsys.readouterr().err
    assert "Remote exec wrapper on barbury-l failed before the command could run" in err
    assert "FileNotFoundError" in err


def test_ucl_exec_sync_json_includes_wrapper_error(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(stdout=exec_stdout(stderr=b"unknown wrapper error\n", returncode=127, wrapper_error=True))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--json", "missing-command"], runner=runner) == 127

    payload = json.loads(capsys.readouterr().out)
    assert payload["wrapper_error"] is True
    assert payload["returncode"] == 127
    assert payload["stderr"] == "unknown wrapper error\n"


def test_ucl_exec_json_is_clean_for_ssh_failure(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(returncode=255, stdout="VBoxManage noise\n", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--json", "hostname"], runner=runner) == 2

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["wrapper_error"] is True
    assert payload["returncode"] == 255
    assert "SSH failed before remote exec wrapper started" in payload["error"]
    assert "VBoxManage" not in payload["stdout"]


def test_ucl_exec_json_reports_command_failure_without_human_text(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return ok(stdout=exec_stdout(stderr=b"bad\n", returncode=7))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--json", "false"], runner=runner) == 7

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["wrapper_error"] is False
    assert payload["stderr"] == "bad\n"


def test_ucl_exec_sync_filters_wrapper_startup_noise_from_errors(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(returncode=255, stdout="", stderr="VBoxManage: noisy startup failure\n")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "VBoxManage" not in err
    assert "no stderr/stdout" in err


def test_ucl_exec_sync_stdin_uses_selected_shell(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def read(self) -> str:
            return "echo hello\n"

    monkeypatch.setattr("sys.stdin", FakeStdin())

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            source = kwargs["input"]
            params = embedded_exec_params(source)
            assert params["mode"] == "stdin"
            assert params["shell"] == "csh"
            assert params["stdin_b64"] == base64.b64encode(b"echo hello\n").decode("ascii")
            return ok(stdout=exec_stdout(stdout=b"hello\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--shell", "csh", "--stdin"], runner=runner) == 0
    assert capsys.readouterr().out == "hello\n"


def test_ucl_exec_stdin_dry_run_reads_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeStdin:
        def read(self) -> str:
            return "echo hello\n"

    monkeypatch.setattr("sys.stdin", FakeStdin())

    rc = main_cli.main(["exec", "barbury-l", "--stdin", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "dry_run: true" in out
    assert "mode:       sync" in out
    assert "shell:      bash" in out


def test_ucl_exec_rejects_bad_command_shapes(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def read(self) -> str:
            return "echo hello\n"

    monkeypatch.setattr("sys.stdin", FakeStdin())

    assert main_cli.main(["exec", "barbury-l"]) == 2
    assert "no remote command provided" in capsys.readouterr().err
    assert main_cli.main(["exec", "barbury-l", "--stdin", "hostname"]) == 2
    assert "--stdin cannot be used with COMMAND" in capsys.readouterr().err
    assert main_cli.main(["exec", "barbury-l", "--session", "work", "hostname"]) == 2
    assert "require --detach" in capsys.readouterr().err
    assert main_cli.main(["exec", "barbury-l", "--unknown", "hostname"]) == 2
    assert "unknown ucl exec option" in capsys.readouterr().err


def test_ucl_exec_detach_preserves_tmux_path(
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
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "mkdir -p" in joined and "tar -xf" not in joined:
            return ok()
        if "cat >" in joined:
            assert "hostname" in kwargs["input"]
            return ok()
        if "tmux new-window" in joined:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--detach", "--", "hostname"], runner=runner) == 0

    assert "session:    work" in capsys.readouterr().out
    assert any("tmux new-window" in " ".join(call) for call in calls)


def test_ucl_exec_detach_requires_explicit_session_when_no_existing_tmux(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout([]))
        raise AssertionError(f"unexpected argv after failed tmux decision: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--detach", "--", "hostname"], runner=runner) == 2

    assert "no tmux sessions exist" in capsys.readouterr().err
    assert not any("cat >" in " ".join(call) for call in calls)


def test_ucl_doctor_reports_host_state(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if argv == remote_python_argv(timeout_seconds=8):
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
        if argv == remote_python_argv():
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


def test_ucl_tail_follow_filters_login_noise(
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

    class FakePipe:
        def __init__(self, lines: list[str] | None = None, text: str = "") -> None:
            self.lines = lines or []
            self.text = text
            self.writes: list[str] = []

        def write(self, value: str) -> None:
            self.writes.append(value)

        def close(self) -> None:
            pass

        def readline(self) -> str:
            return self.lines.pop(0) if self.lines else ""

        def read(self) -> str:
            return self.text

    class FakeFollowPopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            assert argv == remote_python_argv()
            self.stdin = FakePipe()
            self.stdout = FakePipe(
                [
                    "VBoxManage: error: startup noise\n",
                    main_cli.TAIL_SENTINEL_BEGIN + "\n",
                    "actual follow line\n",
                ]
            )
            self.stderr = FakePipe(text="VBoxManage: error: stderr noise\n")

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--follow"], runner=runner, popener=FakeFollowPopen) == 0

    captured = capsys.readouterr()
    assert captured.out == "actual follow line\n"
    assert "VBoxManage" not in captured.out
    assert "VBoxManage" not in captured.err


def test_ucl_fetch_filters_login_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    write_record(
        RunRecord(
            run_id="demo",
            kind="run",
            host="barbury-l",
            ssh_host="barbury-l",
            session="demo",
            window="run",
            remote_dir="/tmp/ucl-machine-tools/launchers/demo",
            log_path="/tmp/ucl-machine-tools/launchers/demo/run.log",
            command=("bash", "run.sh"),
        )
    )
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        data = b"ok\n"
        info = tarfile.TarInfo("run.log")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    encoded = base64.b64encode(tar_buf.getvalue()).decode("ascii")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv():
            assert "/tmp/ucl-machine-tools/launchers/demo" in kwargs["input"]
            return ok(
                stdout="\n".join(
                    [
                        "VBoxManage: error: startup noise",
                        main_cli.FETCH_SENTINEL_BEGIN,
                        encoded,
                        main_cli.FETCH_SENTINEL_END,
                    ]
                )
                + "\n"
            )
        if argv[:2] == ["tar", "-xf"]:
            assert kwargs["input"].startswith(b"././@PaxHeader") or kwargs["input"]
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["fetch", "demo", "--output-dir", str(tmp_path / "out")], runner=runner) == 0

    captured = capsys.readouterr()
    assert "VBoxManage" not in captured.out
    assert "VBoxManage" not in captured.err
    assert str(tmp_path / "out") in captured.out


def test_ucl_clean_filters_login_noise(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv():
            assert "DAYS=7" in kwargs["input"]
            return ok(
                stdout="\n".join(
                    [
                        "VBoxManage: error: startup noise",
                        main_cli.CLEAN_SENTINEL_BEGIN,
                        json.dumps({"schema_version": 1, "paths": ["/tmp/ucl-machine-tools/launchers/old"]}),
                        main_cli.CLEAN_SENTINEL_END,
                    ]
                )
                + "\n"
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["clean", "barbury-l"], runner=runner) == 0

    captured = capsys.readouterr()
    assert captured.out == "/tmp/ucl-machine-tools/launchers/old\n"
    assert "VBoxManage" not in captured.out
    assert "VBoxManage" not in captured.err


def test_ucl_jobs_info_and_stop_use_registry_and_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    write_record(
        RunRecord(
            run_id="demo",
            kind="run",
            host="barbury-l",
            ssh_host="barbury-l",
            session="demo",
            window="run",
            remote_dir="/tmp/ucl-machine-tools/launchers/demo",
            log_path="/tmp/ucl-machine-tools/launchers/demo/run.log",
            command=("bash", "run.sh"),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["demo"]))
        if argv == remote_python_argv() and "kill-window" in kwargs.get("input", ""):
            return ok(stdout='{"returncode": 0, "stdout": "", "stderr": ""}\n')
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["jobs", "--json"], runner=runner) == 0
    jobs_payload = json.loads(capsys.readouterr().out)
    assert jobs_payload["jobs"][0]["status"] == "running"

    assert main_cli.main(["info", "demo", "--json"], runner=runner) == 0
    info_payload = json.loads(capsys.readouterr().out)
    assert info_payload["run_id"] == "demo"
    assert info_payload["status"] == "running"

    assert main_cli.main(["stop", "demo", "--json"], runner=runner) == 0
    stop_payload = json.loads(capsys.readouterr().out)
    assert stop_payload["returncode"] == 0
    assert stop_payload["session"] == "demo"


def test_ucl_stop_requires_explicit_ref_or_yes_for_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    write_record(
        RunRecord(
            run_id="demo",
            kind="run",
            host="barbury-l",
            ssh_host="barbury-l",
            session="demo",
            window="run",
            remote_dir="/tmp/ucl-machine-tools/launchers/demo",
            log_path="/tmp/ucl-machine-tools/launchers/demo/run.log",
            command=("bash", "run.sh"),
        )
    )

    assert main_cli.main(["stop"]) == 2
    assert "run_ref" in capsys.readouterr().err

    calls: list[list[str]] = []
    stop_payload_seen = False

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        nonlocal stop_payload_seen
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv() and "kill-window" in kwargs.get("input", ""):
            stop_payload_seen = True
            return ok(stdout='{"returncode": 0, "stdout": "", "stderr": ""}\n')
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "last"], runner=runner) == 2
    assert "refusing to stop 'last' without --yes" in capsys.readouterr().err
    assert calls == []

    assert main_cli.main(["stop", "last", "--yes", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "demo"
    assert stop_payload_seen is True


def test_ucl_copy_dry_run_and_size_verify(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")

    assert main_cli.main(["copy", str(src), str(dst), "--dry-run", "--partial", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True
    assert "--partial" in dry["argv"]

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        assert argv[0] == "rsync"
        (dst / "a.txt").write_text("hello", encoding="utf-8")
        return ok(stdout="copied\n", stderr="VBoxManage: noisy login\n")

    assert main_cli.main(["copy", str(src), str(dst), "--verify", "size", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["verify"]["ok"] is True
    assert "VBoxManage" not in payload["stderr"]


def test_ucl_env_json_parses_remote_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv():
            assert "/tmp/ucl-machine-tools/fpt" in kwargs["input"]
            return ok(
                stdout="\n".join(
                    [
                        envcheck.ENV_BEGIN,
                        json.dumps(
                            {
                                "schema_version": 1,
                                "remote_root": "/tmp/ucl-machine-tools/fpt",
                                "root_exists": True,
                                "root_created": False,
                                "tmp_free_gb": 500,
                                "cuda_visibility_script": "/usr/local/cuda/CUDA_VISIBILITY.csh",
                                "cuda_visibility_exists": True,
                                "python_setup_script": "/opt/Python/Python-3.11.5_Setup.csh",
                                "python_setup_exists": True,
                                "gpu": None,
                                "gpu_info": None,
                                "ok": True,
                                "errors": [],
                            }
                        ),
                        envcheck.ENV_END,
                    ]
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(["env", "barbury-l", "--remote-root", "/tmp/ucl-machine-tools/fpt", "--json"], runner=runner)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host"] == "barbury-l"
    assert payload["root_exists"] is True


def test_ucl_fanout_runs_command_in_catalog_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv[-2:] == ["python3", "-"]:
            host = argv[-3]
            return ok(stdout=exec_stdout(stdout=f"{host}\n".encode()))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        ["fanout", "--hosts", "barbury-l", "canada-l", "--catalog", str(catalog), "--json", "--", "hostname"],
        runner=runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in payload["results"]] == ["barbury-l", "canada-l"]
    assert [row["stdout"] for row in payload["results"]] == ["barbury-l\n", "canada-l\n"]


def test_generated_remote_python_sources_compile() -> None:
    compile(
        main_cli._sync_exec_source({"mode": "command", "argv": ["hostname"], "timeout": 60.0}),
        "<exec-source>",
        "exec",
    )
    compile(main_cli._tail_source("/tmp/demo/run.log", 20), "<tail-source>", "exec")
    compile(main_cli._tail_follow_source("/tmp/demo/run.log", 20), "<tail-follow-source>", "exec")
    compile(main_cli._fetch_source("/tmp/demo"), "<fetch-source>", "exec")
    compile(main_cli._clean_source(7, False), "<clean-source>", "exec")
    compile(main_cli._clean_source(7, True), "<clean-source-execute>", "exec")
    compile(envcheck.env_source(remote_root="/tmp/ucl-machine-tools/fpt", create=False, gpu=None), "<env-source>", "exec")
    compile(copy_tools.manifest_source("/tmp/demo", sha256=False), "<copy-manifest-source>", "exec")


def test_help_exposes_unified_commands_and_not_legacy_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "status" in help_text
    assert "exec" in help_text
    assert "Common use:" in help_text
    assert "ucl exec barbury-l df -h /tmp" in help_text
    assert "ucl exec barbury-l --detach -- hostname" in help_text
    assert "ucl run --host barbury-l --gpu auto" in help_text
    assert "Use 'ucl COMMAND --help'" in help_text
    assert "ucl-inventory" not in help_text
    assert "ucl-launch" not in help_text
