from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import main_cli


_FRAMING_HINTS = ("frame", "marker", "sentinel", "magic", "ucl_rsync")


def _unexpected_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
    raise AssertionError(f"no command should run: {argv}")


def _python_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    if normalized and Path(normalized[0]).name == "env":
        normalized = normalized[1:]
    assert normalized, "framed transport command must not be empty"
    assert Path(normalized[0]).name.startswith("python"), normalized
    return normalized


def _assert_framed_python_invocation(argv: list[str], *, self_contained: bool = False) -> None:
    assert argv[0] != "ssh", "raw ssh bypasses the framed transport"
    normalized = _python_argv(argv)

    if "-m" in normalized:
        module = normalized[normalized.index("-m") + 1]
        assert not self_contained, "source hosts must not need ucl-machine-tools installed"
        assert "rsync_transport" in module
        return

    assert "-c" in normalized, normalized
    source = normalized[normalized.index("-c") + 1]
    lowered = source.lower()
    assert "ssh" in lowered
    assert any(hint in lowered for hint in _FRAMING_HINTS), "inline wrapper does not expose a framing protocol"
    if self_contained:
        assert "ucl_machine_tools" not in source


def _rsync_transport(argv: list[str]) -> str:
    index = argv.index("-e")
    return argv[index + 1]


def _assert_framed_rsync(argv: list[str], *, self_contained: bool = False) -> None:
    transport = _rsync_transport(argv)
    assert transport != "ssh -o BatchMode=yes -o LogLevel=ERROR"
    _assert_framed_python_invocation(shlex.split(transport), self_contained=self_contained)


def _embedded_rsync_argv(outer: list[str]) -> list[str]:
    for index, token in enumerate(outer[:-2]):
        if Path(token).name not in {"bash", "sh"} or outer[index + 1] not in {"-c", "-lc"}:
            continue
        command = outer[index + 2]
        for _ in range(3):
            parsed = shlex.split(command)
            if len(parsed) == 1 and parsed[0] != command:
                command = parsed[0]
                continue
            assert parsed and parsed[0] == "rsync", parsed
            return parsed

    rsync_index = outer.index("rsync")
    return outer[rsync_index:]


def _manifest(*, contains_payload: bool) -> dict[str, Any]:
    files = []
    if contains_payload:
        files.append(
            {
                "path": "payload.bin",
                "bytes": 7,
                "kind": "file",
                "mode": 0o640,
                "mtime_ns": 1_700_000_000_000_000_000,
                "uid": 1000,
                "gid": 1000,
                "sha256": "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5",
            }
        )
    return {
        "schema_version": 1,
        "exists": True,
        "root_kind": "directory",
        "file_count": len(files),
        "total_bytes": 7 if files else 0,
        "files": files,
        "unsupported": [],
        "empty_directories": [],
    }


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (copy_tools.Endpoint("local-src", None, "/tmp/src"), copy_tools.Endpoint("local-dst", None, "/tmp/dst")),
        (
            copy_tools.Endpoint("remote-src", "source.internal", "/tmp/src"),
            copy_tools.Endpoint("local-dst", None, "/tmp/dst"),
        ),
        (
            copy_tools.Endpoint("local-src", None, "/tmp/src"),
            copy_tools.Endpoint("remote-dst", "destination.internal", "/tmp/dst"),
        ),
    ],
    ids=("local", "download", "upload"),
)
def test_build_rsync_argv_always_uses_framed_python_transport(
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
) -> None:
    argv = copy_tools.build_rsync_argv(src, dst)

    _assert_framed_rsync(argv)


def test_selective_rsync_inherits_framed_transport() -> None:
    argv = copy_tools.build_selective_rsync_argv(
        copy_tools.Endpoint("src", None, "/tmp/src"),
        copy_tools.Endpoint("dst", "destination.internal", "/tmp/dst"),
        source_is_directory=True,
        partial=True,
    )

    assert "--files-from=-" in argv
    _assert_framed_rsync(argv)


def test_every_verified_copy_retry_uses_framed_transport(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _manifest(contains_payload=True)
    missing = _manifest(contains_payload=False)
    destination_reads = iter((missing, source))

    monkeypatch.setattr(
        main_cli,
        "_read_reconcile_manifests",
        lambda *args, **kwargs: {"source": source, "destination": missing},
    )

    def fake_endpoint_manifest(
        endpoint: copy_tools.Endpoint,
        *,
        verify: str,
        runner: Any,
    ) -> dict[str, Any]:
        del verify, runner
        if endpoint.path == "/tmp/dst":
            return next(destination_reads)
        return source

    monkeypatch.setattr(main_cli, "_copy_endpoint_manifest", fake_endpoint_manifest)
    transfers: list[list[str]] = []

    def fake_rsync(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        assert kwargs["input"] == "payload.bin\0"
        transfers.append(argv)
        return SimpleNamespace(returncode=23 if len(transfers) == 1 else 0, stdout="", stderr="")

    rc = main_cli.main(
        [
            "copy",
            "/tmp/src",
            "/tmp/dst",
            "--verify",
            "sha256",
            "--retries",
            "1",
            "--json",
        ],
        runner=fake_rsync,
    )

    assert rc == 0
    assert len(transfers) == 2
    for argv in transfers:
        assert "--files-from=-" in argv
        _assert_framed_rsync(argv)
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["attempts"]) == 2
    for attempt in payload["attempts"]:
        _assert_framed_rsync(attempt["argv"])


@pytest.mark.parametrize("selective", [False, True], ids=("plain", "selective"))
def test_remote_to_remote_outer_hop_uses_framed_transport(selective: bool) -> None:
    src = copy_tools.Endpoint("src", "source.internal", "/tmp/src")
    dst = copy_tools.Endpoint("dst", "destination.internal", "/tmp/dst")
    if selective:
        argv = copy_tools.build_selective_remote_to_remote_argv(
            src,
            dst,
            source_is_directory=True,
        )
    else:
        argv = copy_tools.build_remote_to_remote_argv(src, dst)

    _assert_framed_python_invocation(argv)


@pytest.mark.parametrize("selective", [False, True], ids=("plain", "selective"))
def test_remote_to_remote_inner_rsync_transport_is_framed_and_self_contained(selective: bool) -> None:
    src = copy_tools.Endpoint("src", "source.internal", "/tmp/src")
    dst = copy_tools.Endpoint("dst", "destination.internal", "/tmp/dst")
    if selective:
        outer = copy_tools.build_selective_remote_to_remote_argv(
            src,
            dst,
            source_is_directory=True,
        )
    else:
        outer = copy_tools.build_remote_to_remote_argv(src, dst)

    inner = _embedded_rsync_argv(outer)
    assert inner[-2:] == ["/tmp/src/" if selective else "/tmp/src", "destination.internal:/tmp/dst/" if selective else "destination.internal:/tmp/dst"]
    _assert_framed_rsync(inner, self_contained=True)


@pytest.mark.parametrize(
    "raw_args",
    [
        ("-e", "ssh"),
        ("-essh",),
        ("-ave", "ssh"),
        ("-avessh",),
        ("--rsh", "ssh"),
        ("--rsh=ssh",),
        ("--rsh=",),
        ("--rsync-path", "sudo rsync"),
        ("--rsync-path=sudo rsync",),
        ("--rsync-path=",),
        ("--old-args",),
        ("--no-s",),
        ("--no-secluded-args",),
        ("--no-protect-args",),
        ("-M", "--old-args"),
        ("-aM--old-args",),
        ("--remote-option", "--old-args"),
        ("--remote-option=--old-args",),
        ("--",),
    ],
)
def test_copy_rejects_raw_rsync_transport_overrides(
    raw_args: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main_cli.main(
        ["copy", "/tmp/src", "/tmp/dst", "--dry-run", "--json", "--", *raw_args],
        runner=_unexpected_runner,
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out or captured.err


def test_copy_keeps_benign_raw_rsync_arguments() -> None:
    raw_args = ["--exclude", "*.pt", "--max-size=10G", "--delete-delay"]

    args = main_cli._parse_copy_argv(["/tmp/src", "/tmp/dst", "--", *raw_args])
    argv = copy_tools.build_rsync_argv(
        copy_tools.parse_endpoint(args.src),
        copy_tools.parse_endpoint(args.dst),
        rsync_args=args.rsync_args,
    )

    assert argv[-len(raw_args) - 3 : -3] == raw_args
    assert argv[-3] == "--"


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (copy_tools.Endpoint("src", None, "/tmp/src"), copy_tools.Endpoint("dst", "remote", "/tmp/dst")),
        (copy_tools.Endpoint("src", "remote", "/tmp/src"), copy_tools.Endpoint("dst", None, "/tmp/dst")),
    ],
)
def test_remote_copy_forces_protected_arguments_after_raw_options(
    src: copy_tools.Endpoint,
    dst: copy_tools.Endpoint,
) -> None:
    argv = copy_tools.build_rsync_argv(src, dst, rsync_args=("--exclude", "*.pt"))

    assert argv.count("--protect-args") == 1
    assert argv.index("--protect-args") < argv.index("--exclude")


@pytest.mark.parametrize("raw_arg", ["-T/tmp/cache", "-fmerge,- *.pt"])
def test_copy_keeps_attached_short_option_values_containing_e(raw_arg: str) -> None:
    copy_tools.validate_rsync_args((raw_arg,))


def test_copy_separates_dash_leading_local_operands_from_options() -> None:
    argv = copy_tools.build_rsync_argv(
        copy_tools.parse_endpoint("./-essh"),
        copy_tools.parse_endpoint("./-destination"),
    )

    assert argv[-3:] == ["--", "./-essh", "./-destination"]


def test_copy_dry_run_json_exposes_framed_transport_for_audit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main_cli.main(
        ["copy", "/tmp/src", "/tmp/dst", "--dry-run", "--json"],
        runner=_unexpected_runner,
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["mode"] == "rsync"
    _assert_framed_rsync(payload["argv"])
