from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import launch
from ucl_machine_tools.hosts import HostSpec


def host() -> HostSpec:
    return HostSpec(name="barbury-l", ssh_host="barbury-l", labels=("ucl-gpu",), restart_policy="lab_pc")


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text("echo hi\n", encoding="utf-8")
    return bundle


def test_bash_and_csh_launchers_have_no_profile_artifacts(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    bash_plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", session="demo")
    bash_launcher, bash_files = launch.build_launcher_files(bash_plan)

    assert bash_launcher == ".ucl_payload.sh"
    assert "echo '[ucl] shell: bash'" in bash_files[".ucl_payload.sh"]
    assert "bash run.sh" in bash_files[".ucl_payload.sh"]

    csh_plan = launch.build_exec_plan(
        host=host(),
        stdin_body="source /opt/Python/Python-3.11.5_Setup.csh\npython3 --version\n",
        shell="csh",
        session="demo_csh",
    )
    csh_launcher, csh_files = launch.build_launcher_files(csh_plan)

    assert csh_launcher == ".ucl_launch.sh"
    assert "echo '[ucl] shell: csh'" in csh_files[".ucl_launch.sh"]
    assert "csh -f /tmp/ucl-machine-tools/launchers/demo_csh/.ucl_payload.csh" in csh_files[".ucl_launch.sh"]
    assert "source /opt/Python/Python-3.11.5_Setup.csh" in csh_files[".ucl_payload.csh"]
    combined = "\n".join((*bash_files.values(), *csh_files.values()))
    for forbidden in ("profile", "uv run", "tsg", "run_prefix", "preflight", "csh-bootstrap", "launch_profiles"):
        assert forbidden not in combined


def test_env_and_gpu_are_plain_exports_not_profiles(tmp_path: Path) -> None:
    plan = launch.build_exec_plan(
        host=host(),
        command=("env",),
        env=(("CUDA_VISIBLE_DEVICES", "0"), ("FOO", "bar baz")),
        session="demo",
    )

    _, files = launch.build_launcher_files(plan)

    assert "export CUDA_VISIBLE_DEVICES=0" in files[".ucl_payload.sh"]
    assert "export FOO='bar baz'" in files[".ucl_payload.sh"]


def test_staged_run_uses_separate_working_directory_and_frozen_uv(tmp_path: Path) -> None:
    plan = launch.build_staged_run_plan(
        host=host(),
        source_dir="/tmp/thakwani/fpt/stages/demo/sources/source123",
        environment_dir="/tmp/thakwani/fpt/stages/demo/envs/env123",
        uv_bin="/tmp/thakwani/fpt/tools/uv/0.9.27/uv",
        uv_cache_dir="/tmp/thakwani/fpt/cache/uv",
        python_install_dir="/tmp/thakwani/fpt/tools/python",
        script="scripts/train.sh",
        args=("--steps", "10"),
        session="demo",
        remote_root="/tmp/thakwani/fpt/launchers",
    )

    assert plan.remote_dir == "/tmp/thakwani/fpt/launchers/demo"
    assert plan.work_dir == "/tmp/thakwani/fpt/stages/demo/sources/source123"
    assert plan.command == (
        "/tmp/thakwani/fpt/tools/uv/0.9.27/uv",
        "run",
        "--frozen",
        "--no-sync",
        "--project",
        "/tmp/thakwani/fpt/stages/demo/sources/source123",
        "--",
        "bash",
        "scripts/train.sh",
        "--steps",
        "10",
    )
    assert ("UV_PROJECT_ENVIRONMENT", "/tmp/thakwani/fpt/stages/demo/envs/env123") in plan.env
    assert ("UV_CACHE_DIR", "/tmp/thakwani/fpt/cache/uv") in plan.env
    assert ("UV_PYTHON_INSTALL_DIR", "/tmp/thakwani/fpt/tools/python") in plan.env

    _, files = launch.build_launcher_files(plan)
    payload = files[".ucl_payload.sh"]
    assert "cd /tmp/thakwani/fpt/stages/demo/sources/source123" in payload
    assert "mkdir -p /tmp/thakwani/fpt/launchers/demo" in payload
    assert "uv run --frozen --no-sync" in payload


@pytest.mark.parametrize("script", ("/absolute.sh", "../escape.sh", "scripts/../../escape.sh", ""))
def test_staged_run_rejects_unsafe_script_paths(script: str) -> None:
    with pytest.raises(ValueError, match="script"):
        launch.build_staged_run_plan(
            host=host(),
            source_dir="/tmp/project",
            environment_dir="/tmp/env",
            uv_bin="/tmp/tools/uv",
            uv_cache_dir="/tmp/cache",
            script=script,
            session="demo",
        )


def test_remote_root_can_be_configured_in_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = make_bundle(tmp_path)

    plan = launch.build_run_plan(
        host=host(),
        local_dir=bundle,
        script="run.sh",
        session="demo",
        remote_root="/tmp/ucl-machine-tools/fpt/launchers",
    )
    assert plan.remote_root == "/tmp/ucl-machine-tools/fpt/launchers"
    assert plan.remote_dir == "/tmp/ucl-machine-tools/fpt/launchers/demo"

    with pytest.raises(ValueError, match="under /tmp/ucl-machine-tools/fpt/launchers"):
        launch.build_exec_plan(
            host=host(),
            command=("hostname",),
            session="demo",
            remote_root="/tmp/ucl-machine-tools/fpt/launchers",
            remote_dir="/tmp/ucl-machine-tools/launchers/demo",
        )

    monkeypatch.setenv("UCL_LAUNCH_ROOT", "/tmp/ucl-machine-tools/env-launchers")
    env_plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", session="envdemo")
    assert env_plan.remote_root == "/tmp/ucl-machine-tools/env-launchers"
    assert env_plan.remote_dir == "/tmp/ucl-machine-tools/env-launchers/envdemo"


def test_tmux_sentinel_parser_ignores_noise_and_exec_auto_requires_single_session() -> None:
    argv = launch.build_tmux_list_argv(host(), timeout_seconds=8)
    assert argv[0] == "python3"
    assert argv[-3:] == ["barbury-l", "python3", "-"]

    stdout = "\n".join(
        [
            "Last login noise",
            launch.TMUX_SENTINEL_BEGIN,
            json.dumps({"schema_version": 1, "sessions": ["work"]}),
            launch.TMUX_SENTINEL_END,
            "Connection closed",
        ]
    )
    assert launch.parse_tmux_sessions(stdout) == ("work",)
    with pytest.raises(ValueError, match="sentinel"):
        launch.parse_tmux_sessions("a\nb\n")

    decision = launch.decide_tmux(
        sessions=("work",),
        generated_session="generated",
        requested_session=None,
        new_session=False,
        window="exec_hostname",
        require_explicit_when_not_single=True,
    )
    assert decision.mode == "new-window"
    assert decision.session == "work"

    with pytest.raises(RuntimeError, match="no tmux sessions"):
        launch.decide_tmux(
            sessions=(),
            generated_session="generated",
            requested_session=None,
            new_session=False,
            window="exec_hostname",
            require_explicit_when_not_single=True,
        )


def test_tmux_session_probe_has_a_local_timeout() -> None:
    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs["timeout"] == 11
        raise subprocess.TimeoutExpired(argv, timeout=kwargs["timeout"])

    with pytest.raises(RuntimeError, match="timed out listing tmux sessions on barbury-l after 11s"):
        launch.list_remote_sessions(host(), runner=runner, timeout_seconds=8)


def test_upload_and_launcher_writes_use_argv_only(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", session="demo")
    runner_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []

    class FakeStdout:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            popen_calls.append(argv)
            self.stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        runner_calls.append(argv)
        assert kwargs.get("shell", False) is False
        if "tar -xf -" in " ".join(argv):
            assert kwargs.get("stdin") is not None
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        assert "input" in kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    launch.upload_bundle(plan, runner=runner, popener=FakePopen)
    launch.write_launcher_files(plan, runner=runner)

    assert popen_calls[0][:2] == ["tar", "-cf"]
    assert all(call[0] == "python3" for call in runner_calls)
    assert all("--logical-argv" in call for call in runner_calls)
    assert "umask 077" in " ".join(runner_calls[-1])
    assert "chmod 700" in " ".join(runner_calls[-1])
    assert "scp" not in " ".join(" ".join(call) for call in runner_calls)
    assert all("rsync" not in call for call in runner_calls)
