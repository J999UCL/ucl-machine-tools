from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools.hosts import HostSpec
from ucl_machine_tools import launch, profiles


def host() -> HostSpec:
    return HostSpec(name="barbury-l", ssh_host="barbury-l", labels=("ucl-gpu",), restart_policy="lab_pc")


def write_profiles(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def profile_file(tmp_path: Path) -> Path:
    return write_profiles(
        tmp_path / "profiles.json",
        {
            "schema_version": 1,
            "profiles": {
                "base": {
                    "env": {"A": "1"},
                    "preflight": [{"label": "base", "command": "echo base"}],
                },
                "uv-a": {
                    "extends": ["base"],
                    "env": {"A": "2", "B": "3"},
                    "run_prefix": ["uv", "run", "--"],
                },
                "tsg": {
                    "shell": "csh-bootstrap",
                    "source": ["/opt/Python/Python-3.11.5_Setup.csh"],
                    "preflight_after_setup": [{"label": "torch", "command": "python3 -c 'print(1)'"}],
                },
            },
        },
    )


def test_profile_loader_resolves_extends_env_and_cli_precedence(tmp_path: Path) -> None:
    catalog = profiles.load_profiles(explicit_files=[profile_file(tmp_path)])

    resolved = profiles.resolve_profiles(["uv-a"], catalog=catalog, cli_env=(("B", "cli"), ("C", "4")))

    assert resolved.names == ("uv-a",)
    assert dict(resolved.env) == {"A": "2", "B": "cli", "C": "4"}
    assert [check.label for check in resolved.preflight] == ["base"]
    assert resolved.run_prefix == ("uv", "run", "--")


def test_profile_validation_rejects_bad_schema_unknown_field_cycle_and_run_prefix_conflict(tmp_path: Path) -> None:
    bad_schema = write_profiles(tmp_path / "bad_schema.json", {"schema_version": 2, "profiles": {}})
    with pytest.raises(ValueError, match="schema_version"):
        profiles.load_profile_file(bad_schema)

    bad_field = write_profiles(tmp_path / "bad_field.json", {"schema_version": 1, "profiles": {"x": {"surprise": True}}})
    with pytest.raises(ValueError, match="unknown fields"):
        profiles.load_profile_file(bad_field)

    cycle = write_profiles(
        tmp_path / "cycle.json",
        {"schema_version": 1, "profiles": {"a": {"extends": ["b"]}, "b": {"extends": ["a"]}}},
    )
    catalog = profiles.load_profiles(explicit_files=[cycle])
    with pytest.raises(ValueError, match="cycle"):
        profiles.resolve_profiles(["a"], catalog=catalog)

    conflict = write_profiles(
        tmp_path / "conflict.json",
        {
            "schema_version": 1,
            "profiles": {
                "a": {"run_prefix": ["uv", "run", "--"]},
                "b": {"run_prefix": ["python", "-m"]},
            },
        },
    )
    catalog = profiles.load_profiles(explicit_files=[conflict])
    with pytest.raises(ValueError, match="run_prefix"):
        profiles.resolve_profiles(["a", "b"], catalog=catalog)


def test_build_run_launcher_plain_uv_and_tsg(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text("echo hi\n", encoding="utf-8")
    catalog = profiles.load_profiles(explicit_files=[profile_file(tmp_path)])

    plain = profiles.resolve_profiles(["plain-bash"], catalog=catalog)
    plain_plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", profile=plain, session="demo")
    launcher_name, files = launch.build_launcher_files(plain_plan)
    assert launcher_name == ".ucl_payload.sh"
    assert "bash run.sh" in files[".ucl_payload.sh"]
    assert "exec > >(tee -a /tmp/ucl-machine-tools/launchers/demo/run.log) 2>&1" in files[".ucl_payload.sh"]

    uv = profiles.resolve_profiles(["uv-a"], catalog=catalog)
    uv_plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", profile=uv, session="demo")
    _, uv_files = launch.build_launcher_files(uv_plan)
    assert "uv run -- bash run.sh" in uv_files[".ucl_payload.sh"]
    assert "echo base" in uv_files[".ucl_payload.sh"]

    tsg = profiles.resolve_profiles(["tsg"], catalog=catalog, cli_env=(("CUDA_VISIBLE_DEVICES", "0"),))
    tsg_plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", profile=tsg, session="demo")
    tsg_launcher, tsg_files = launch.build_launcher_files(tsg_plan)
    assert tsg_launcher == ".ucl_launch.csh"
    assert "source /opt/Python/Python-3.11.5_Setup.csh" in tsg_files[".ucl_launch.csh"]
    assert "setenv CUDA_VISIBLE_DEVICES 0" in tsg_files[".ucl_launch.csh"]
    assert "python3 -c 'print(1)'" in tsg_files[".ucl_payload.sh"]


def test_exec_stdin_with_run_prefix_uses_heredoc(tmp_path: Path) -> None:
    catalog = profiles.load_profiles(explicit_files=[profile_file(tmp_path)])
    uv = profiles.resolve_profiles(["uv-a"], catalog=catalog)
    plan = launch.build_exec_plan(host=host(), profile=uv, stdin_body="echo hello\n", session="demo")

    _, files = launch.build_launcher_files(plan)

    assert "uv run -- bash <<'UCL_STDIN_SCRIPT'" in files[".ucl_payload.sh"]
    assert "echo hello" in files[".ucl_payload.sh"]


def test_tmux_sentinel_parser_ignores_noise_and_exec_auto_requires_single_session() -> None:
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


def test_upload_and_launcher_writes_use_argv_only(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text("echo hi\n", encoding="utf-8")
    profile = profiles.resolve_profiles(["plain-bash"], catalog=profiles.load_profiles())
    plan = launch.build_run_plan(host=host(), local_dir=bundle, script="run.sh", profile=profile, session="demo")
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
    assert all(call[0] == "ssh" for call in runner_calls)
    assert "scp" not in " ".join(" ".join(call) for call in runner_calls)
    assert "rsync" not in " ".join(" ".join(call) for call in runner_calls)
