from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import launch
from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools import launch_cli


def host() -> HostSpec:
    return HostSpec(name="barbury-l", ssh_host="barbury-l", labels=("ucl-gpu",), restart_policy="lab_pc")


def ok(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def fail(stdout: str = "", stderr: str = "failed") -> SimpleNamespace:
    return SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr)


def make_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    script = bundle / "run.sh"
    script.write_text("#!/usr/bin/env bash\necho hello\n", encoding="utf-8")
    return bundle


def test_build_plan_validates_local_inputs_and_remote_dir(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    plan = launch.build_plan(host=host(), local_dir=bundle, script="run.sh", session="demo")

    assert plan.script_rel == "run.sh"
    assert plan.remote_dir == "/tmp/ucl-machine-tools/launchers/demo"

    with pytest.raises(ValueError, match="local_dir"):
        launch.build_plan(host=host(), local_dir=tmp_path / "missing", script="run.sh")
    with pytest.raises(ValueError, match="script"):
        launch.build_plan(host=host(), local_dir=bundle, script="missing.sh")
    outside = tmp_path / "outside.sh"
    outside.write_text("echo no\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside local_dir"):
        launch.build_plan(host=host(), local_dir=bundle, script=str(outside))
    with pytest.raises(ValueError, match="remote_dir must be absolute"):
        launch.build_plan(host=host(), local_dir=bundle, script="run.sh", remote_dir="relative")
    with pytest.raises(ValueError, match="under /tmp/ucl-machine-tools/launchers"):
        launch.build_plan(host=host(), local_dir=bundle, script="run.sh", remote_dir="/tmp/elsewhere")


def test_generated_launcher_contains_cd_env_args_and_log(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    plan = launch.build_plan(
        host=host(),
        local_dir=bundle,
        script="run.sh",
        session="demo",
        args=["--x", "two words"],
        env=["FOO=bar baz"],
    )

    source = launch.build_launcher_source(plan)

    assert "cd /tmp/ucl-machine-tools/launchers/demo" in source
    assert "export FOO='bar baz'" in source
    assert "bash run.sh --x 'two words'" in source
    assert "tee -a /tmp/ucl-machine-tools/launchers/demo/run.log" in source


@pytest.mark.parametrize(
    ("sessions", "requested", "new_session", "mode", "session"),
    [
        ((), None, False, "new-session", "demo"),
        (("existing",), None, False, "new-window", "existing"),
        (("existing",), "existing", False, "new-window", "existing"),
        ((), "missing", False, "new-session", "missing"),
        ((), "fresh", True, "new-session", "fresh"),
    ],
)
def test_tmux_decision_modes(
    sessions: tuple[str, ...],
    requested: str | None,
    new_session: bool,
    mode: str,
    session: str,
) -> None:
    decision = launch.decide_tmux(
        sessions=sessions,
        generated_session="demo",
        requested_session=requested,
        new_session=new_session,
        window="run",
    )

    assert decision.mode == mode
    assert decision.session == session


def test_tmux_decision_rejects_multiple_auto_and_existing_new_session() -> None:
    with pytest.raises(RuntimeError, match="multiple tmux sessions"):
        launch.decide_tmux(
            sessions=("a", "b"),
            generated_session="demo",
            requested_session=None,
            new_session=False,
            window="run",
        )
    with pytest.raises(RuntimeError, match="already exists"):
        launch.decide_tmux(
            sessions=("demo",),
            generated_session="demo",
            requested_session="demo",
            new_session=True,
            window="run",
        )


def test_upload_uses_tar_over_ssh_and_not_scp_or_rsync(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    plan = launch.build_plan(host=host(), local_dir=bundle, script="run.sh", session="demo")
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
        assert kwargs.get("stdin") is not None
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    launch.upload_bundle(plan, runner=runner, popener=FakePopen)

    assert popen_calls[0][:2] == ["tar", "-cf"]
    assert runner_calls[0][0] == "ssh"
    flat = " ".join(runner_calls[0])
    assert "scp" not in flat
    assert "rsync" not in flat


def test_write_launcher_and_tmux_launch_use_argv_only(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    plan = launch.build_plan(host=host(), local_dir=bundle, script="run.sh", session="demo")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        assert kwargs.get("shell", False) is False
        return ok()

    launch.write_launcher(plan, runner=runner)
    decision = launch.decide_tmux(sessions=(), generated_session="demo", requested_session="demo", new_session=True, window="run")
    launch.launch_tmux(plan, decision, runner=runner)

    assert ".ucl_launch.sh" in " ".join(calls[0][0])
    assert "bash run.sh" in calls[0][1]["input"]
    assert "tmux new-session" in " ".join(calls[1][0])


def test_launch_cli_dry_run_does_not_call_remote_runner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = make_bundle(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise AssertionError("dry-run must not call runner")

    rc = launch_cli.main(
        ["--host", "barbury-l", "--local-dir", str(bundle), "--script", "run.sh", "--dry-run"],
        runner=runner,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "dry_run: true" in out
    assert "upload local dir with tar over SSH" in out


def test_launch_cli_full_fake_path_reuses_single_existing_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = make_bundle(tmp_path)
    calls: list[list[str]] = []

    class FakeStdout:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            calls.append(argv)
            self.stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        joined = " ".join(argv)
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            assert ".ucl_launch.sh" in joined
            assert "bash run.sh hello" in kwargs["input"]
            return ok()
        if "tmux list-sessions" in joined:
            return ok(stdout="work\n")
        if "tmux new-window" in joined:
            return ok()
        raise AssertionError(f"unexpected argv: {argv}")

    rc = launch_cli.main(
        ["--host", "barbury-l", "--local-dir", str(bundle), "--script", "run.sh", "--arg", "hello"],
        runner=runner,
        popener=FakePopen,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "session:    work" in out
    assert "tmux attach -t work" in out
    assert "tail -f" in out
    assert any("tmux new-window" in " ".join(call) for call in calls)


def test_launch_cli_multiple_sessions_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle = make_bundle(tmp_path)
    calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            calls.append(argv)
            self.stdout = None

        def wait(self) -> int:
            return 0

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        joined = " ".join(argv)
        if "tar -xf -" in joined:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "cat >" in joined:
            return ok()
        if "tmux list-sessions" in joined:
            return ok(stdout="a\nb\n")
        return ok()

    rc = launch_cli.main(
        ["--host", "barbury-l", "--local-dir", str(bundle), "--script", "run.sh"],
        runner=runner,
        popener=FakePopen,
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "multiple tmux sessions" in err
    assert "a, b" in err
    assert not any(call and call[0] == "tar" for call in calls)
    assert not any("tar -xf -" in " ".join(call) for call in calls)


def test_launch_help_has_no_use_master(capsys: pytest.CaptureFixture[str]) -> None:
    parser = launch_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])

    assert "--use-master" not in capsys.readouterr().out
