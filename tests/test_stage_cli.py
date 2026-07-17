from __future__ import annotations

from pathlib import Path

import pytest

from ucl_machine_tools import main_cli


def test_stage_parser_exposes_explicit_uv_workflow(tmp_path: Path) -> None:
    args = main_cli.build_parser().parse_args(
        [
            "stage",
            "--uv",
            "--host",
            "barbury-l",
            "--name",
            "fpt",
            "--local-dir",
            str(tmp_path),
            "--remote-root",
            "/tmp/thakwani/fpt",
            "--gpu",
            "auto",
            "--dry-run",
        ]
    )

    assert args.command == "stage"
    assert args.uv is True
    assert args.host == "barbury-l"
    assert args.name == "fpt"
    assert args.local_dir == tmp_path
    assert args.remote_root == "/tmp/thakwani/fpt"
    assert args.gpu == "auto"


def test_run_parser_accepts_stage_without_host_or_local_dir() -> None:
    args = main_cli.build_parser().parse_args(
        ["run", "--stage", "fpt-barbury-l-abcd1234", "--script", "scripts/train.sh", "--new-session"]
    )

    assert args.stage == "fpt-barbury-l-abcd1234"
    assert args.host is None
    assert args.local_dir is None


@pytest.mark.parametrize(
    "argv, message",
    (
        (
            ["run", "--stage", "ready", "--host", "barbury-l", "--script", "run.sh", "--new-session"],
            "--stage cannot be combined with --host",
        ),
        (
            ["run", "--stage", "ready", "--local-dir", ".", "--script", "run.sh", "--new-session"],
            "--stage cannot be combined with --local-dir",
        ),
        (
            ["run", "--host", "barbury-l", "--script", "run.sh", "--new-session"],
            "ordinary ucl run requires --local-dir",
        ),
        (
            ["run", "--local-dir", ".", "--script", "run.sh", "--new-session"],
            "ordinary ucl run requires --host",
        ),
    ),
)
def test_run_mode_validation_is_explicit(argv: list[str], message: str) -> None:
    args = main_cli.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        main_cli._validate_run_mode(args)


def test_top_level_help_mentions_stage_then_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_cli.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "ucl stage --uv" in help_text
    assert "ucl run --stage STAGE_ID" in help_text

