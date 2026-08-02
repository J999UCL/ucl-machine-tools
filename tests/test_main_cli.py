from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import envcheck, job_control, launch, main_cli
from ucl_machine_tools import ssh as ssh_tools
from ucl_machine_tools.registry import RunRecord, read_record, write_record


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


def job_identity(
    *,
    exists: bool = True,
    session: str = "demo",
    window: str = "run",
    boot_id: str = "boot-123",
    tmux_socket_path: str = "/tmp/tmux-1/default",
    tmux_server_pid: int = 4321,
    pane_id: str = "%7",
    window_id: str = "@3",
    pane_pid: int = 1234,
    start_ticks: int | None = 5678,
    session_id: int | None = 1234,
    pane_dead: bool = False,
    terminal_at_capture: bool | None = None,
) -> dict[str, Any]:
    identity = {
        "exists": exists,
        "session": session,
        "window": window,
        "boot_id": boot_id,
        "tmux_socket_path": tmux_socket_path,
        "tmux_server_pid": tmux_server_pid,
        "pane_id": pane_id,
        "window_id": window_id,
        "pane_pid": pane_pid,
        "pane_start_ticks": start_ticks,
        "pane_session_id": session_id,
        "pane_dead": pane_dead,
        "pane_dead_status": None,
    }
    if terminal_at_capture is not None:
        identity["terminal_at_capture"] = terminal_at_capture
    return identity


def identity_stdout(identity: dict[str, Any], *, ok_result: bool = True, error: str = "") -> str:
    return "\n".join(
        [
            "login noise",
            job_control.IDENTITY_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "ok": ok_result, "identity": identity, "error": error}),
            job_control.IDENTITY_SENTINEL_END,
        ]
    )


def launch_stdout(identity: dict[str, Any], *, ok_result: bool = True, error: str = "") -> str:
    return "\n".join(
        [
            "login noise",
            job_control.LAUNCH_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "ok": ok_result, "identity": identity, "error": error}),
            job_control.LAUNCH_SENTINEL_END,
        ]
    )


def stop_stdout(*, status: str = "stopped", ok_result: bool = True, **extra: Any) -> str:
    payload = {
        "schema_version": 1,
        "ok": ok_result,
        "status": status,
        "target": "demo:run",
        "signal": "TERM",
        "expected_identity": job_identity(),
        "current_identity": {"exists": False},
        "signal_errors": [],
        "survivors": [],
        "cleanup": "target_already_gone",
        **extra,
    }
    return "\n".join([job_control.STOP_SENTINEL_BEGIN, json.dumps(payload), job_control.STOP_SENTINEL_END])


def remote_python_argv(host: str = "barbury-l", *, timeout_seconds: int | None = None) -> list[str]:
    return ssh_tools.build_remote_python_argv(host, timeout_seconds=timeout_seconds)


def inventory_stdout(
    host: str = "barbury-l",
    *,
    busy: bool = False,
    gpus: list[dict[str, Any]] | None = None,
) -> str:
    gpu = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "memory_total_mb": 24576,
        "memory_used_mb": 1024,
        "memory_free_mb": 23552,
        "utilization_gpu_percent": 65 if busy else 1,
        "processes": [{"pid": 7}] if busy else [],
    }
    payload = {
        "schema_version": 1,
        "host": host,
        "hostname": host,
        "ok": True,
        "gpus": [gpu] if gpus is None else gpus,
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


def test_ucl_status_human_output_streams_rows_in_completion_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canada = main_cli.inventory.parse_sentinel_stdout(inventory_stdout("canada-l", busy=True))
    canada["status"] = "busy"
    barbury = main_cli.inventory.parse_sentinel_stdout(inventory_stdout("barbury-l"))
    barbury["status"] = "ready"
    completed = [canada, barbury]

    def fake_collect(hosts: Any, *, on_result: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert on_result is not None
        for row in completed:
            on_result(row)
        return list(reversed(completed))

    monkeypatch.setattr(main_cli.inventory, "collect", fake_collect)
    monkeypatch.setattr(main_cli, "ensure_knuckles_master", lambda **kwargs: "existing")

    assert main_cli.main(["status", "barbury-l", "canada-l"]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0].startswith("host")
    assert lines[1].startswith("canada-l")
    assert lines[2].startswith("barbury-l")


def test_ucl_status_uses_fast_parallel_defaults() -> None:
    args = main_cli.build_parser().parse_args(["status", "all"])

    assert args.jobs == 32
    assert args.timeout_seconds == 5


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
    assert sorted(probed) == ["barbury-l", "canada-l"]


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
    assert sorted(calls) == ["barbury-l", "canada-l"]

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
    assert sorted(calls) == ["barbury-l", "canada-l"]

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
    assert sorted(probed) == ["barbury-l", "canada-l", "cream"]

    probed.clear()
    assert main_cli.main(["status", "barbury-l", "3090ti", "--catalog", str(catalog), "--json"], runner=runner) == 0
    dedupe_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in dedupe_payload["hosts"]] == ["barbury-l", "canada-l"]
    assert sorted(probed) == ["barbury-l", "canada-l"]

    probed.clear()
    assert main_cli.main(["status", "all", "!cream", "--catalog", str(catalog), "--json"], runner=runner) == 0
    exclusion_payload = json.loads(capsys.readouterr().out)
    assert [row["host"] for row in exclusion_payload["hosts"]] == ["barbury-l", "canada-l"]
    assert sorted(probed) == ["barbury-l", "canada-l"]


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
    assert sorted(probed) == ["barbury-l", "canada-l", "cream"]


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
        if argv == remote_python_argv(timeout_seconds=8) and "UCL_INVENTORY_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=inventory_stdout())
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            assert "exec > >(tee -a" in kwargs["input"]
            assert "export CUDA_VISIBLE_DEVICES=0" in kwargs["input"]
            assert "export SECRET_TOKEN=abc" in kwargs["input"]
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=launch_stdout(job_identity(session="work", window="run")))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        [
            "run",
            "--host",
            "barbury-l",
            "--gpu",
            "auto",
            "--min-free-vram-gb",
            "22",
            "--env",
            "SECRET_TOKEN=abc",
            "--project",
            "fpt",
            "--session",
            "work",
            "--local-dir",
            str(bundle),
            "--script",
            "run.sh",
            "--arg",
            "x",
        ],
        runner=runner,
        popener=FakePopen,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "session:    work" in out
    latest = tmp_path / "cache" / "runs" / "latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8"))["run_id"] == "work"
    record = read_record("last")
    assert record.kind == "run"
    assert record.identity == job_identity(session="work", window="run")
    provenance = record.provenance
    assert provenance["project"] == "fpt"
    assert provenance["selected_gpu"] == "0"
    assert provenance["bundle_path"] == str(bundle.resolve())
    expected_script_sha = hashlib.sha256((bundle / "run.sh").read_bytes()).hexdigest()
    assert provenance["script_sha256"] == expected_script_sha
    assert "local_git_sha" in provenance
    assert "CUDA_VISIBLE_DEVICES" in provenance["env_keys"]
    assert "SECRET_TOKEN" in provenance["env_keys"]
    assert provenance["env"]["SECRET_TOKEN"] == "<redacted>"
    assert "abc" not in json.dumps(provenance)


def test_ucl_run_gpu_auto_injects_busy_gpu_when_vram_meets_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    bundle = make_bundle(tmp_path)
    launcher_sources: list[str] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and "UCL_INVENTORY_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=inventory_stdout(busy=True))
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            launcher_sources.append(kwargs["input"])
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=launch_stdout(job_identity(session="work", window="run")))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        [
            "run",
            "--host",
            "barbury-l",
            "--gpu",
            "auto",
            "--min-free-vram-gb",
            "22",
            "--session",
            "work",
            "--local-dir",
            str(bundle),
            "--script",
            "run.sh",
        ],
        runner=runner,
        popener=FakePopen,
    )

    assert rc == 0
    assert launcher_sources
    assert "export CUDA_VISIBLE_DEVICES=0" in launcher_sources[0]


def test_ucl_run_requires_explicit_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = make_bundle(tmp_path)

    rc = main_cli.main(["run", "--host", "barbury-l", "--local-dir", str(bundle), "--script", "run.sh"])

    assert rc == 2
    assert "ucl run requires --session NAME or --new-session" in capsys.readouterr().err


def test_ucl_run_writes_provisional_record_before_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    bundle = make_bundle(tmp_path)
    launch_attempted = False

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        nonlocal launch_attempted
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            launch_attempted = True
            return ok(stdout="launch completed without protocol output")
        raise AssertionError(f"unexpected argv: {argv}")

    assert (
        main_cli.main(
            [
                "run",
                "--host",
                "barbury-l",
                "--project",
                "fpt",
                "--session",
                "work",
                "--local-dir",
                str(bundle),
                "--script",
                "run.sh",
            ],
            runner=runner,
            popener=FakePopen,
        )
        == 2
    )

    assert launch_attempted is True
    record = read_record("last")
    assert record.kind == "run"
    assert record.session == "work"
    assert record.identity == {"pending_launch": True}
    assert record.provenance["project"] == "fpt"
    assert record.provenance["script_sha256"] == hashlib.sha256((bundle / "run.sh").read_bytes()).hexdigest()


def test_run_registry_backfills_missing_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    runs = tmp_path / "cache" / "runs"
    runs.mkdir(parents=True)
    payload = {
        "run_id": "old",
        "kind": "run",
        "host": "barbury-l",
        "ssh_host": "barbury-l",
        "session": "old",
        "window": "run",
        "remote_dir": "/tmp/ucl-machine-tools/launchers/old",
        "log_path": "/tmp/ucl-machine-tools/launchers/old/run.log",
        "command": ["bash", "run.sh"],
    }
    (runs / "old.json").write_text(json.dumps(payload), encoding="utf-8")

    record = read_record("old")

    assert record.command == ("bash", "run.sh")
    assert record.provenance == {}


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


def test_ucl_exec_normalizes_explicit_bash_login_shell_without_filtering_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv("dopey", timeout_seconds=30):
            params = embedded_exec_params(kwargs["input"])
            assert params["argv"] == [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                "printf command-output",
            ]
            return ok(stdout=exec_stdout(stdout=b"command-output", stderr=b""))
        raise AssertionError(f"unexpected argv: {argv}")

    assert (
        main_cli.main(
            ["exec", "dopey", "--", "bash", "-lc", "printf command-output"],
            runner=runner,
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == "command-output"
    assert captured.err == ""


def test_ucl_exec_sync_accepts_multiple_hosts_with_delimiter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = write_status_catalog(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv[-2:] == ["python3", "-"]:
            assert "--handshake-timeout" in argv
            assert argv[argv.index("--handshake-timeout") + 1] == "30.0"
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
        if argv == remote_python_argv(timeout_seconds=8):
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
        [
            "exec",
            "barbury-l",
            "--gpu",
            "auto",
            "--min-free-vram-gb",
            "22",
            "--cwd",
            "/tmp",
            "--timeout",
            "1",
            "--json",
            "pwd",
        ],
        runner=runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stdout"] == "/tmp\n"
    assert payload["stderr"] == ""
    assert payload["returncode"] == 0
    assert payload["timed_out"] is False


def test_ucl_exec_gpu_auto_selects_most_free_vram_despite_busy_signals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    busy_gpu = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "memory_total_mb": 24576,
        "memory_used_mb": 1024,
        "memory_free_mb": 23552,
        "utilization_gpu_percent": 65,
        "processes": [{"pid": 7}],
    }
    clean_gpu = {
        "index": 1,
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "memory_total_mb": 24576,
        "memory_used_mb": 2048,
        "memory_free_mb": 22528,
        "utilization_gpu_percent": 1,
        "processes": [],
    }

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            return ok(stdout=inventory_stdout(gpus=[busy_gpu, clean_gpu]))
        if argv == remote_python_argv(timeout_seconds=30):
            params = embedded_exec_params(kwargs["input"])
            assert params["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            return ok(stdout=exec_stdout(stdout=b"0\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        ["exec", "barbury-l", "--gpu", "auto", "--min-free-vram-gb", "20", "printenv", "CUDA_VISIBLE_DEVICES"],
        runner=runner,
    )

    assert rc == 0
    assert capsys.readouterr().out == "0\n"


def test_ucl_exec_gpu_auto_respects_min_free_vram_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            return ok(stdout=inventory_stdout())
        raise AssertionError(f"unexpected argv after failed GPU selection: {argv}")

    rc = main_cli.main(["exec", "barbury-l", "--gpu", "auto", "--min-free-vram-gb", "24", "--json", "pwd"], runner=runner)

    assert rc == 2
    assert "no free GPU found on barbury-l" in capsys.readouterr().err


def test_ucl_exec_gpu_auto_allows_shared_gpu_with_enough_free_vram(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = main_cli.inventory.parse_sentinel_stdout(inventory_stdout())
    payload["gpus"][0]["processes"] = [{"pid": 834627, "user": "other", "used_memory_mb": 1024}]
    payload["gpus"][0]["memory_free_mb"] = 20 * 1024
    payload["gpus"][0]["utilization_gpu_percent"] = 95
    inventory_result = "\n".join(
        (
            "login noise",
            main_cli.inventory.INVENTORY_SENTINEL_BEGIN,
            json.dumps(payload),
            main_cli.inventory.INVENTORY_SENTINEL_END,
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            return ok(stdout=inventory_result)
        if argv == remote_python_argv(timeout_seconds=30):
            params = embedded_exec_params(kwargs["input"])
            assert params["env"]["CUDA_VISIBLE_DEVICES"] == "0"
            return ok(stdout=exec_stdout(stdout=b"shared-ok\n"))
        raise AssertionError(f"unexpected argv: {argv}")

    assert (
        main_cli.main(
            ["exec", "barbury-l", "--gpu", "auto", "--min-free-vram-gb", "20", "--json", "hostname"],
            runner=runner,
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["stdout"] == "shared-ok\n"


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


def test_ucl_exec_json_preserves_post_frame_failure_output(capsys: pytest.CaptureFixture[str]) -> None:
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
    assert payload["stdout"] == "VBoxManage noise\n"


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


def test_ucl_exec_sync_preserves_post_frame_wrapper_errors(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=30):
            return SimpleNamespace(returncode=255, stdout="", stderr="VBoxManage: noisy startup failure\n")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "hostname"], runner=runner) == 2

    err = capsys.readouterr().err
    assert "VBoxManage: noisy startup failure" in err
    assert "no stderr/stdout" not in err


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


def test_ucl_run_and_detached_exec_accept_remote_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = make_bundle(tmp_path)

    assert (
        main_cli.main(
            [
                "run",
                "--host",
                "barbury-l",
                "--local-dir",
                str(bundle),
                "--script",
                "run.sh",
                "--remote-root",
                "/tmp/ucl-machine-tools/fpt/launchers",
                "--new-session",
                "--dry-run",
            ]
        )
        == 0
    )
    run_out = capsys.readouterr().out
    assert "remote_dir: /tmp/ucl-machine-tools/fpt/launchers/run_" in run_out
    assert "remote_root: /tmp/ucl-machine-tools/fpt/launchers" in run_out

    class FakeStdin:
        def read(self) -> str:
            return "echo hello\n"

    monkeypatch.setattr("sys.stdin", FakeStdin())
    assert (
        main_cli.main(
            [
                "exec",
                "barbury-l",
                "--detach",
                "--stdin",
                "--remote-root",
                "/tmp/ucl-machine-tools/fpt/launchers",
                "--dry-run",
            ]
        )
        == 0
    )
    exec_out = capsys.readouterr().out
    assert "remote_dir: /tmp/ucl-machine-tools/fpt/launchers/exec_stdin_" in exec_out
    assert "remote_root: /tmp/ucl-machine-tools/fpt/launchers" in exec_out


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
    assert main_cli.main(["exec", "barbury-l", "--min-free-vram-gb", "-1", "hostname"]) == 2
    assert "--min-free-vram-gb must be >= 0" in capsys.readouterr().err
    assert main_cli.main(["exec", "barbury-l", "--min-free-vram-gb", "lots", "hostname"]) == 2
    assert "--min-free-vram-gb must be a number" in capsys.readouterr().err


def test_ucl_exec_detach_preserves_tmux_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    calls: list[list[str]] = []
    launch_sources: list[str] = []

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
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            launch_sources.append(kwargs["input"])
            return ok(stdout=launch_stdout(job_identity(session="work", window="exec_hostname")))
        raise AssertionError(f"unexpected argv: {argv}")

    assert (
        main_cli.main(
            [
                "exec",
                "barbury-l",
                "--detach",
                "--gpu",
                "1",
                "--env",
                "SECRET_TOKEN=abc",
                "--project",
                "smoke",
                "--",
                "hostname",
            ],
            runner=runner,
        )
        == 0
    )

    assert "session:    work" in capsys.readouterr().out
    assert launch_sources and '"tmux", "new-window"' in launch_sources[0]
    record = read_record("last")
    assert record.identity == job_identity(session="work", window="exec_hostname")
    provenance = record.provenance
    assert provenance["project"] == "smoke"
    assert provenance["selected_gpu"] == "1"
    assert provenance["bundle_path"] == ""
    assert provenance["script_sha256"] == ""
    assert "local_git_sha" in provenance
    assert provenance["env"]["SECRET_TOKEN"] == "<redacted>"
    assert "abc" not in json.dumps(provenance)


def test_ucl_exec_detach_writes_provisional_record_before_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "mkdir -p" in joined and "tar -xf" not in joined:
            return ok()
        if "cat >" in joined:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout="not a launch sentinel")
        raise AssertionError(f"unexpected argv: {argv}")

    assert (
        main_cli.main(
            [
                "exec",
                "barbury-l",
                "--detach",
                "--project",
                "smoke",
                "--",
                "hostname",
            ],
            runner=runner,
        )
        == 2
    )

    record = read_record("last")
    assert record.kind == "exec"
    assert record.session == "work"
    assert record.identity == {"pending_launch": True}
    assert record.provenance["project"] == "smoke"


def test_ucl_exec_detach_finalizes_a_fast_terminal_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))
    terminal_identity = job_identity(
        exists=False,
        pane_dead=True,
        start_ticks=None,
        session_id=1234,
        terminal_at_capture=True,
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        joined = " ".join(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv() and "UCL_TMUX_JSON_BEGIN" in kwargs.get("input", ""):
            return ok(stdout=tmux_stdout(["work"]))
        if "mkdir -p" in joined and "tar -xf" not in joined:
            return ok()
        if "cat >" in joined:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.LAUNCH_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=launch_stdout(terminal_identity))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["exec", "barbury-l", "--detach", "--", "hostname"], runner=runner) == 0
    record = read_record("last")
    assert record.identity == terminal_identity
    assert record.identity.get("pending_launch") is None


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


def test_ucl_doctor_reports_unreachable_host_without_tmux_discovery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and "UCL_INVENTORY_JSON_BEGIN" in kwargs.get("input", ""):
            return SimpleNamespace(
                returncode=255,
                stdout="",
                stderr="ssh: connect to host barbury-l: No route to host",
            )
        raise AssertionError(f"unexpected call after unreachable inventory probe: {argv}")

    assert main_cli.main(["doctor", "barbury-l"], runner=runner) == 2

    out = capsys.readouterr().out
    assert "host:          barbury-l" in out
    assert "status:        unreachable" in out
    assert "tmux_sessions: unavailable" in out
    assert "error:         target host is unreachable from the jump host" in out
    assert "no-sentinel" not in out
    assert len(calls) == 2
    assert not any("UCL_TMUX_JSON_BEGIN" in kwargs.get("input", "") for _, kwargs in calls)


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


@pytest.mark.parametrize("flag", ["--live", "--follow"])
def test_ucl_tail_parser_exposes_live_mode(flag: str) -> None:
    args = main_cli.build_parser().parse_args(["tail", "demo", flag])

    assert args.command == "tail"
    assert args.run_ref == "demo"
    assert args.live is True


@pytest.mark.parametrize("live_flag", ["--live", "--follow"])
def test_ucl_tail_live_streams_and_filters_only_startup_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    live_flag: str,
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

        def read(self, size: int = -1) -> str:
            text, self.text = self.text, ""
            return text

    class FakeFollowPopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            assert argv == remote_python_argv()
            self.stdin = FakePipe()
            self.stdout = FakePipe(
                [
                    "VBoxManage: error: startup noise\n",
                    main_cli.TAIL_SENTINEL_BEGIN + "\n",
                    "actual follow line\n",
                    "actual log mentions VirtualBox\n",
                ]
            )
            self.stderr = FakePipe(text="VBoxManage from the remote command\n")

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", live_flag], runner=runner, popener=FakeFollowPopen) == 0

    captured = capsys.readouterr()
    assert captured.out == "actual follow line\nactual log mentions VirtualBox\n"
    assert captured.err == "VBoxManage from the remote command\n"


def test_ucl_tail_help_documents_live_streaming(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_cli.main(["tail", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "--live, --follow" in help_text
    assert "until interrupted with Ctrl-C" in help_text


def test_live_tail_stderr_forwarder_keeps_draining_after_output_failure() -> None:
    class Stream:
        def __init__(self) -> None:
            self.chunks = ["first", "second", ""]
            self.reads = 0

        def read(self, size: int) -> str:
            self.reads += 1
            return self.chunks.pop(0)

    class BrokenDestination:
        def write(self, chunk: str) -> None:
            raise BrokenPipeError("local stderr closed")

        def flush(self) -> None:
            pass

    stream = Stream()
    read_errors: list[bool] = []

    with pytest.raises(BrokenPipeError, match="local stderr closed"):
        main_cli._forward_text_stream(stream, BrokenDestination(), on_read_error=lambda: read_errors.append(True))

    assert stream.reads == 3
    assert read_errors == []


def test_live_tail_stderr_read_failure_triggers_remote_termination() -> None:
    class BrokenStream:
        def read(self, size: int) -> str:
            raise OSError("remote stderr read failed")

    terminated: list[bool] = []

    with pytest.raises(OSError, match="remote stderr read failed"):
        main_cli._forward_text_stream(BrokenStream(), io.StringIO(), on_read_error=lambda: terminated.append(True))

    assert terminated == [True]


def test_ucl_tail_live_streams_incrementally_without_filtering_log_content(
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
    output_before_next_line: list[str] = []

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
            if self.lines and self.lines[0].startswith("training log mentions"):
                output_before_next_line.append(capsys.readouterr().out)
            return self.lines.pop(0) if self.lines else ""

        def read(self, size: int = -1) -> str:
            text, self.text = self.text, ""
            return text

    class FakeLivePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            assert argv == remote_python_argv()
            self.stdin = FakePipe()
            self.stdout = FakePipe(
                [
                    "Last login: wrapper noise\n",
                    "VBoxManage: error: startup noise\n",
                    main_cli.TAIL_SENTINEL_BEGIN + "\n",
                    "first live log line\n",
                    "training log mentions VBoxManage as data\n",
                ]
            )
            self.stderr = FakePipe()

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--live"], runner=runner, popener=FakeLivePopen) == 0

    captured = capsys.readouterr()
    assert output_before_next_line == ["first live log line\n"]
    assert captured.out == "training log mentions VBoxManage as data\n"
    assert captured.err == ""


def test_ucl_tail_live_drains_stderr_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import threading

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
    stderr_read_started = threading.Event()

    class InputPipe:
        def write(self, value: str) -> None:
            pass

        def close(self) -> None:
            pass

    class StdoutPipe:
        def __init__(self) -> None:
            self.lines = [main_cli.TAIL_SENTINEL_BEGIN + "\n", "live output\n"]

        def readline(self) -> str:
            assert stderr_read_started.wait(timeout=1), "stderr was not drained concurrently"
            return self.lines.pop(0) if self.lines else ""

    class StderrPipe:
        def __init__(self) -> None:
            self.emitted = False

        def read(self, size: int = -1) -> str:
            stderr_read_started.set()
            if self.emitted:
                return ""
            self.emitted = True
            return "remote stderr\n"

    class FakeLivePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            assert argv == remote_python_argv()
            self.stdin = InputPipe()
            self.stdout = StdoutPipe()
            self.stderr = StderrPipe()

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            pass

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--live"], runner=runner, popener=FakeLivePopen) == 0

    captured = capsys.readouterr()
    assert captured.out == "live output\n"
    assert captured.err == "remote stderr\n"


def test_ucl_tail_live_keyboard_interrupt_terminates_and_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    class InputPipe:
        def write(self, value: str) -> None:
            pass

        def close(self) -> None:
            pass

    class StdoutPipe:
        def readline(self) -> str:
            raise KeyboardInterrupt

    class StderrPipe:
        def read(self, size: int = -1) -> str:
            return ""

    class FakeLivePopen:
        instance: "FakeLivePopen"

        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            self.stdin = InputPipe()
            self.stdout = StdoutPipe()
            self.stderr = StderrPipe()
            self.terminated = False
            self.killed = False
            self.wait_timeouts: list[float | None] = []
            FakeLivePopen.instance = self

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            return 143

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--live"], runner=runner, popener=FakeLivePopen) == 130
    assert FakeLivePopen.instance.terminated is True
    assert FakeLivePopen.instance.killed is False
    assert FakeLivePopen.instance.wait_timeouts == [main_cli.TAIL_STOP_TIMEOUT_SECONDS]


def test_live_tail_stop_escalates_to_kill_after_timeout() -> None:
    class Process:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_timeouts: list[float | None] = []

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("ucl tail", timeout)
            return 137

    proc = Process()
    main_cli._terminate_and_wait(proc)

    assert proc.terminated is True
    assert proc.killed is True
    assert proc.wait_timeouts == [main_cli.TAIL_STOP_TIMEOUT_SECONDS, None]


def test_ucl_tail_live_propagates_nonzero_ssh_exit(
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

        def write(self, value: str) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> str:
            return self.lines.pop(0) if self.lines else ""

        def read(self, size: int = -1) -> str:
            text, self.text = self.text, ""
            return text

    class FakeFailedLivePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            assert argv == remote_python_argv()
            self.stdin = FakePipe()
            self.stdout = FakePipe(["Last login: wrapper noise\n"])
            self.stderr = FakePipe(text="ssh: connect to host barbury-l: No route to host\n")

        def wait(self) -> int:
            return 255

        def terminate(self) -> None:
            pass

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["tail", "demo", "--live"], runner=runner, popener=FakeFailedLivePopen) == 255

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ssh: connect to host barbury-l: No route to host\n"


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
            provenance={"project": "fpt", "selected_gpu": "0"},
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


def test_ucl_clean_uses_configured_remote_root(capsys: pytest.CaptureFixture[str]) -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv():
            assert 'ROOT="/tmp/ucl-machine-tools/fpt/launchers"' in kwargs["input"]
            return ok(
                stdout="\n".join(
                    [
                        main_cli.CLEAN_SENTINEL_BEGIN,
                        json.dumps({"schema_version": 1, "paths": []}),
                        main_cli.CLEAN_SENTINEL_END,
                    ]
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["clean", "barbury-l", "--remote-root", "/tmp/ucl-machine-tools/fpt/launchers"], runner=runner) == 0
    assert capsys.readouterr().out == ""


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
            provenance={"project": "fpt", "selected_gpu": "0"},
            identity=job_identity(),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.IDENTITY_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=identity_stdout(job_identity()))
        if argv == remote_python_argv(timeout_seconds=8) and job_control.STOP_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=stop_stdout())
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["jobs", "--json"], runner=runner) == 0
    jobs_payload = json.loads(capsys.readouterr().out)
    assert jobs_payload["jobs"][0]["status"] == "running"
    assert jobs_payload["jobs"][0]["project"] == "fpt"
    assert jobs_payload["jobs"][0]["selected_gpu"] == "0"

    assert main_cli.main(["info", "demo", "--json"], runner=runner) == 0
    info_payload = json.loads(capsys.readouterr().out)
    assert info_payload["run_id"] == "demo"
    assert info_payload["status"] == "running"
    assert info_payload["provenance"]["project"] == "fpt"

    assert main_cli.main(["stop", "demo", "--json"], runner=runner) == 0
    stop_payload = json.loads(capsys.readouterr().out)
    assert stop_payload["status"] == "stopped"
    assert stop_payload["ok"] is True
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
            identity=job_identity(),
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
        if argv == remote_python_argv(timeout_seconds=8) and job_control.STOP_SENTINEL_BEGIN in kwargs.get("input", ""):
            stop_payload_seen = True
            return ok(stdout=stop_stdout())
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "last"], runner=runner) == 2
    assert "refusing to stop 'last' without --yes" in capsys.readouterr().err
    assert calls == []

    assert main_cli.main(["stop", "last", "--yes", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "demo"
    assert stop_payload_seen is True


@pytest.mark.parametrize(
    ("recorded_identity", "current_identity", "expected_status"),
    [
        ({}, job_identity(), "legacy_unverified"),
        (job_identity(), job_identity(pane_pid=9999), "identity_mismatch"),
        (job_identity(), {"exists": False, "session": "demo", "window": "run"}, "exited_or_missing"),
    ],
)
def test_ucl_info_uses_exact_window_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recorded_identity: dict[str, Any],
    current_identity: dict[str, Any],
    expected_status: str,
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
            identity=recorded_identity,
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.IDENTITY_SENTINEL_BEGIN in kwargs.get("input", ""):
            return ok(stdout=identity_stdout(current_identity))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["info", "demo", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status
    assert payload["identity"] == recorded_identity
    assert payload["current_identity"] == current_identity


@pytest.mark.parametrize(
    ("status", "ok_result", "extra", "expected_rc"),
    [
        ("already_stopped", True, {}, 0),
        ("identity_mismatch", False, {}, 2),
        ("legacy_unverified", False, {}, 2),
        (
            "still_running",
            False,
            {
                "survivors": [{"pid": 1234, "pgrp": 1234, "start_ticks": 5678}],
                "signal_errors": [{"pgrp": 1234, "error": "PermissionError"}],
            },
            2,
        ),
    ],
)
def test_ucl_stop_reports_structured_handled_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    ok_result: bool,
    extra: dict[str, Any],
    expected_rc: int,
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
            identity=job_identity() if status != "legacy_unverified" else {},
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.STOP_SENTINEL_BEGIN in kwargs.get("input", ""):
            assert "kill-session" not in kwargs["input"]
            return ok(stdout=stop_stdout(status=status, ok_result=ok_result, **extra))
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "demo", "--json"], runner=runner) == expected_rc
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert payload["ok"] is ok_result
    assert payload["survivors"] == extra.get("survivors", [])
    assert payload["signal_errors"] == extra.get("signal_errors", [])


@pytest.mark.parametrize(
    ("failure_kind", "expected_status", "expected_error"),
    [
        ("timeout", "unknown_after_timeout", "timed out"),
        ("ssh_nonzero", "wrapper_error", "connection refused"),
        ("missing_sentinel", "wrapper_error", "sentinel"),
        ("malformed_sentinel", "wrapper_error", "valid JSON"),
    ],
)
def test_ucl_stop_json_always_returns_one_object_for_wrapper_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_kind: str,
    expected_status: str,
    expected_error: str,
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
            identity=job_identity(),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8) and job_control.STOP_SENTINEL_BEGIN in kwargs.get("input", ""):
            if failure_kind == "timeout":
                raise main_cli.subprocess.TimeoutExpired(argv, kwargs["timeout"])
            if failure_kind == "ssh_nonzero":
                return SimpleNamespace(returncode=255, stdout="", stderr="ssh: Connection refused\n")
            if failure_kind == "missing_sentinel":
                return ok(stdout="VBoxManage: login noise\nremote wrapper vanished\n")
            return ok(
                stdout="\n".join(
                    [
                        job_control.STOP_SENTINEL_BEGIN,
                        "{not-json",
                        job_control.STOP_SENTINEL_END,
                    ]
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "demo", "--json"], runner=runner) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["status"] == expected_status
    assert payload["wrapper_error"] is True
    assert expected_error.lower() in payload["error"].lower()
    assert payload["run_id"] == "demo"


def test_ucl_stop_json_preserves_structured_helper_error(
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
            identity=job_identity(),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            return ok(
                stdout=stop_stdout(
                    status="helper_error",
                    ok_result=False,
                    error="RuntimeError: process inspection failed",
                    cleanup_error="tmux query failed",
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "demo", "--json"], runner=runner) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["status"] == "helper_error"
    assert payload["error"] == "RuntimeError: process inspection failed"
    assert payload["cleanup_error"] == "tmux query failed"


def test_ucl_stop_json_handles_missing_registry_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UCL_MACHINE_TOOLS_CACHE", str(tmp_path / "cache"))

    assert main_cli.main(["stop", "missing", "--json"]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["ok"] is False
    assert payload["status"] == "wrapper_error"
    assert payload["run_id"] == "missing"
    assert "not found" in payload["error"]


@pytest.mark.parametrize(
    ("probe_kind", "expected_status"),
    [("ssh", "unreachable"), ("protocol", "probe_error")],
)
def test_ucl_info_distinguishes_unreachable_hosts_from_probe_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    probe_kind: str,
    expected_status: str,
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
            identity=job_identity(),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            if probe_kind == "ssh":
                return SimpleNamespace(returncode=255, stdout="", stderr="connection refused")
            return ok(stdout="not a sentinel")
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["info", "demo", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == expected_status


def test_ucl_stop_human_output_reports_cleanup_error(
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
            identity=job_identity(),
        )
    )

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == remote_python_argv(timeout_seconds=8):
            return ok(
                stdout=stop_stdout(
                    status="cleanup_failed",
                    ok_result=False,
                    cleanup_error="can't remove pane",
                )
            )
        raise AssertionError(f"unexpected argv: {argv}")

    assert main_cli.main(["stop", "demo"], runner=runner) == 2
    captured = capsys.readouterr()
    assert "status: cleanup_failed" in captured.out
    assert "cleanup_error: can't remove pane" in captured.err


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
        shutil.copy2(src / "a.txt", dst / "a.txt")
        return ok(stdout="copied\n", stderr="VBoxManage: legitimate rsync diagnostic\n")

    assert main_cli.main(["copy", str(src), str(dst), "--verify", "size", "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["verify"]["ok"] is True
    assert payload["stderr"] == "VBoxManage: legitimate rsync diagnostic\n"


def test_ucl_plain_copy_preserves_noise_like_post_handshake_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        assert argv[0] == "rsync"
        return ok(
            stdout="VirtualBox is ordinary rsync stdout\n",
            stderr="VBoxManage is ordinary rsync stderr\n",
        )

    assert main_cli.main(["copy", str(src), str(dst), "--json"], runner=runner) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stdout"] == "VirtualBox is ordinary rsync stdout\n"
    assert payload["stderr"] == "VBoxManage is ordinary rsync stderr\n"


def test_ucl_copy_passes_raw_rsync_args_after_delimiter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    rc = main_cli.main(
        [
            "copy",
            str(src),
            str(dst),
            "--dry-run",
            "--progress",
            "--json",
            "--",
            "--exclude",
            "*.pt",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "rsync"
    assert payload["argv"] == [
        "rsync",
        "-a",
        "--human-readable",
        "-e",
        copy_tools.RSYNC_SSH,
        "--info=progress2",
        "--dry-run",
            "--exclude",
            "*.pt",
            "--dry-run",
            "--",
            str(src),
            str(dst),
    ]


def test_ucl_copy_rejects_unknown_wrapper_options_before_delimiter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    rc = main_cli.main(["copy", str(src), str(dst), "--delete"])

    assert rc == 2
    assert "unknown ucl copy option: --delete" in capsys.readouterr().err


def test_ucl_copy_resolves_aliases_and_rejects_multi_host_selectors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "groups": {"3090ti": ["barbury-l", "canada-l"]},
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "aliases": ["barb"], "gpu_class": "3090ti"},
                    "canada-l": {"ssh": "canada.internal", "gpu_class": "3090ti"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main_cli.main(
        [
            "copy",
            "barb:/tmp/src",
            "canada-l:/tmp/dst",
            "--catalog",
            str(catalog),
            "--dry-run",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved_src"] == "barbury.internal:/tmp/src"
    assert payload["resolved_dst"] == "canada.internal:/tmp/dst"

    def no_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise AssertionError(f"unexpected runner call: {argv}")

    rc = main_cli.main(
        ["copy", "3090ti:/tmp/src", str(tmp_path / "dst"), "--catalog", str(catalog), "--dry-run"],
        runner=no_runner,
    )

    assert rc == 2
    assert "copy endpoint selector must resolve to exactly one host" in capsys.readouterr().err


def test_ucl_copy_remote_to_remote_runs_rsync_from_source_host(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "aliases": ["barb"], "gpu_class": "3090ti"},
                    "barnacle-l": {"ssh": "barnacle.internal", "aliases": ["barn"], "gpu_class": "3090ti"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent-test.sock")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh-add", "-l"]:
            return ok(stdout="256 SHA256:test controller-key\n")
        if argv == copy_tools.build_remote_destination_probe_argv("barbury.internal", "barnacle.internal"):
            return ok()
        if argv == remote_python_argv("barnacle.internal"):
            return ok(stdout=copy_tools.PRESENCE_BEGIN + "\n{" + '"exists": false' + "}\n" + copy_tools.PRESENCE_END + "\n")
        assert argv[0] == "python3"
        assert "barbury.internal" in argv
        inner = argv[argv.index("rsync") :]
        assert inner[:5] == ["rsync", "-a", "--human-readable", "-e", copy_tools.RSYNC_SSH]
        assert inner[5:8] == ["--partial", "--protect-args", "--exclude"]
        assert inner[8] == "*.pt"
        assert inner[-3] == "--"
        assert inner[-2:] == ["/tmp/src ' path", "barnacle.internal:/tmp/dst ' path"]
        return ok(stdout="copied\n")

    rc = main_cli.main(
        [
            "copy",
            "barb:/tmp/src ' path",
            "barn:/tmp/dst ' path",
            "--catalog",
            str(catalog),
            "--partial",
            "--json",
            "--",
            "--exclude",
            "*.pt",
        ],
        runner=runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "remote-to-remote"
    assert payload["ok"] is False
    assert payload["agent_forwarding"] is True
    assert payload["authentication_preflight"] == "passed"
    assert payload["verify"]["message"] == "destination path does not exist after rsync"
    assert not any(call[0] == "rsync" for call in calls)


def test_ucl_copy_remote_to_remote_requires_a_local_agent_before_remote_work(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "gpu_class": "3090ti"},
                    "dopey": {"ssh": "dopey", "gpu_class": "temporary"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)

    def unexpected_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise AssertionError(f"agent preflight should fail locally: {argv}")

    rc = main_cli.main(
        [
            "copy",
            "barbury-l:/tmp/src",
            "dopey:/tmp/dst",
            "--catalog",
            str(catalog),
            "--json",
        ],
        runner=unexpected_runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "SSH_AUTH_SOCK" in payload["error"]
    assert payload["agent_forwarding"] is True
    assert payload["authentication_preflight"] == "failed"


def test_ucl_copy_remote_to_remote_rejects_an_agent_with_no_identities(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "gpu_class": "3090ti"},
                    "dopey": {"ssh": "dopey", "gpu_class": "temporary"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent-test.sock")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv == ["ssh-add", "-l"]:
            return SimpleNamespace(returncode=1, stdout="The agent has no identities.\n", stderr="")
        raise AssertionError(f"remote work must not start: {argv}")

    rc = main_cli.main(
        [
            "copy",
            "barbury-l:/tmp/src",
            "dopey:/tmp/dst",
            "--catalog",
            str(catalog),
            "--json",
        ],
        runner=runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "no loaded identities" in payload["error"]
    assert calls == [["ssh-add", "-l"]]


def test_ucl_copy_remote_to_remote_auth_probe_blocks_rsync(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "gpu_class": "3090ti"},
                    "dopey": {"ssh": "dopey", "gpu_class": "temporary"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent-test.sock")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv == ["ssh-add", "-l"]:
            return ok(stdout="256 SHA256:test controller-key\n")
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == copy_tools.build_remote_destination_probe_argv("barbury.internal", "dopey"):
            return SimpleNamespace(returncode=255, stdout="", stderr="Permission denied (publickey).")
        if argv[0] == "python3":
            raise AssertionError("rsync must not start after destination auth failure")
        raise AssertionError(f"unexpected command: {argv}")

    rc = main_cli.main(
        [
            "copy",
            "barbury-l:/tmp/src",
            "dopey:/tmp/dst",
            "--catalog",
            str(catalog),
            "--json",
        ],
        runner=runner,
    )

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "could not authenticate" in payload["error"]
    assert "Permission denied" in payload["error"]
    assert payload["authentication_preflight"] == "failed"
    assert not any("rsync" in call for call in calls)


def test_ucl_copy_remote_to_remote_verify_reads_each_endpoint_host(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "ucl_hosts.json"
    catalog.write_text(
        json.dumps(
            {
                "hosts": {
                    "barbury-l": {"ssh": "barbury.internal", "gpu_class": "3090ti"},
                    "barnacle-l": {"ssh": "barnacle.internal", "gpu_class": "3090ti"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent-test.sock")
    manifest_hosts: list[str] = []
    transfer_hosts: list[str] = []

    def manifest_stdout() -> str:
        payload = {"schema_version": 1, "exists": True, "file_count": 1, "total_bytes": 5, "files": [{"path": "a.txt", "bytes": 5}]}
        return "\n".join([copy_tools.MANIFEST_BEGIN, json.dumps(payload), copy_tools.MANIFEST_END])

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        if argv == ["ssh-add", "-l"]:
            return ok(stdout="256 SHA256:test controller-key\n")
        if argv == copy_tools.build_remote_destination_probe_argv("barbury.internal", "barnacle.internal"):
            return ok()
        if argv == remote_python_argv("barbury.internal"):
            manifest_hosts.append("barbury.internal")
            assert "/tmp/src" in kwargs["input"]
            return ok(stdout=manifest_stdout())
        if argv == remote_python_argv("barnacle.internal"):
            manifest_hosts.append("barnacle.internal")
            assert "/tmp/dst" in kwargs["input"]
            return ok(stdout=manifest_stdout())
        if argv[:7] == ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "barbury.internal"]:
            transfer_hosts.append("barbury.internal")
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    rc = main_cli.main(
        [
            "copy",
            "barbury-l:/tmp/src",
            "barnacle-l:/tmp/dst",
            "--catalog",
            str(catalog),
            "--verify",
            "size",
            "--json",
        ],
        runner=runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["agent_forwarding"] is True
    assert payload["authentication_preflight"] == "passed"
    assert payload["verify"]["ok"] is True
    assert sorted(manifest_hosts) == ["barbury.internal", "barbury.internal", "barnacle.internal"]
    assert transfer_hosts == []


def test_ucl_copy_verify_failure_cases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.txt").write_text("hello", encoding="utf-8")

    def failed_rsync(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv[0] == "rsync"
        return SimpleNamespace(returncode=23, stdout="", stderr="rsync failed\n")

    rc = main_cli.main(["copy", str(src), str(dst), "--verify", "size", "--json"], runner=failed_rsync)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["verify"]["ok"] is False
    assert payload["returncode"] == 23

    def mismatch_rsync(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv[0] == "rsync"
        (dst / "a.txt").write_text("different-size", encoding="utf-8")
        return ok()

    rc = main_cli.main(["copy", str(src), str(dst), "--verify", "size", "--json"], runner=mismatch_rsync)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["verify"]["ok"] is False
    assert "total_bytes differs" in payload["verify"]["message"]

    dst.joinpath("a.txt").unlink()

    def sha_mismatch_rsync(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert argv[0] == "rsync"
        (dst / "a.txt").write_text("HELLO", encoding="utf-8")
        return ok()

    rc = main_cli.main(["copy", str(src), str(dst), "--verify", "sha256", "--json"], runner=sha_mismatch_rsync)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["verify"]["ok"] is False
    assert payload["verify"]["message"] == "sha256 manifest differs"


def test_copy_endpoint_parsing_edge_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    local_colon = copy_tools.parse_endpoint("/tmp/path:with-colon")
    assert local_colon.host is None
    assert local_colon.path == "/tmp/path:with-colon"

    home_local = copy_tools.parse_endpoint("~/local")
    assert home_local.host is None
    assert home_local.path == str(tmp_path / "local")

    with pytest.raises(ValueError, match="remote endpoint path must be absolute"):
        copy_tools.parse_endpoint("host:relative")
    with pytest.raises(ValueError, match="remote endpoint path must be absolute"):
        copy_tools.parse_endpoint("host:")
    with pytest.raises(ValueError, match="remote endpoint path must be absolute"):
        copy_tools.parse_endpoint("host:~/x")
    with pytest.raises(ValueError, match="unsafe remote selector"):
        copy_tools.parse_endpoint("bad host:/tmp/x")


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


def test_generated_remote_python_sources_compile() -> None:
    compile(
        main_cli._sync_exec_source({"mode": "command", "argv": ["hostname"], "timeout": 60.0}),
        "<exec-source>",
        "exec",
    )
    compile(main_cli._tail_source("/tmp/demo/run.log", 20), "<tail-source>", "exec")
    compile(main_cli._tail_follow_source("/tmp/demo/run.log", 20), "<tail-follow-source>", "exec")
    compile(main_cli._fetch_source("/tmp/demo"), "<fetch-source>", "exec")
    compile(main_cli._clean_source(7, False, "/tmp/ucl-machine-tools/launchers"), "<clean-source>", "exec")
    compile(main_cli._clean_source(7, True, "/tmp/ucl-machine-tools/launchers"), "<clean-source-execute>", "exec")
    compile(envcheck.env_source(remote_root="/tmp/ucl-machine-tools/fpt", create=False, gpu=None), "<env-source>", "exec")
    compile(copy_tools.manifest_source("/tmp/demo", sha256=False), "<copy-manifest-source>", "exec")


def test_help_exposes_unified_commands_and_not_legacy_scripts(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "status" in help_text
    assert "exec" in help_text
    assert "Common workflows:" in help_text
    assert "Inspect machines:" in help_text
    assert "Run quick synchronous commands:" in help_text
    assert "Launch and manage tmux-backed jobs:" in help_text
    assert "Copy data:" in help_text
    assert "ucl exec barbury-l df -h /tmp" in help_text
    assert "ucl exec barbury-l --detach --new-session -- hostname" in help_text
    assert "ucl run --host barbury-l --new-session --gpu auto" in help_text
    assert "ucl copy barbury-l:/tmp/a barnacle-l:/tmp/a -- --partial" in help_text
    assert "Use 'ucl COMMAND --help'" in help_text
    assert "fanout" not in help_text
    assert "ucl-inventory" not in help_text
    assert "ucl-launch" not in help_text


def test_exec_help_mentions_multiple_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    parser = main_cli._build_exec_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "HOST_OR_SELECTOR" in help_text
    assert "[HOST_OR_SELECTOR ...] -- COMMAND" in help_text
    assert "ucl exec barbury-l canada-l barnacle-l -- hostname" in help_text
