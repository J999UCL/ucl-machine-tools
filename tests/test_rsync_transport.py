from __future__ import annotations

import base64
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ucl_machine_tools import copy as copy_tools
from ucl_machine_tools import rsync_transport


FAKE_SSH = Path(__file__).parent / "helpers" / "fake_ssh.py"


def _encoded(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _transport_env(*, stdout_prefix: bytes = b"", stderr_prefix: bytes = b"", mode: str = "run") -> dict[str, str]:
    return {
        **os.environ,
        "FAKE_SSH_STDOUT_PREFIX_B64": _encoded(stdout_prefix),
        "FAKE_SSH_STDERR_PREFIX_B64": _encoded(stderr_prefix),
        "FAKE_SSH_MODE": mode,
    }


def test_prefix_frame_strips_only_bytes_before_split_marker_then_becomes_transparent() -> None:
    frame = rsync_transport.PrefixFrame(b"<ready>", max_prefix_bytes=32)

    assert frame.feed(b"login noise<rea") == b""
    assert frame.feed(b"dy>\x00\xffpayload") == b"\x00\xffpayload"
    assert frame.ready is True
    assert frame.prefix == b"login noise"
    assert frame.feed(b"<ready>is now ordinary data") == b"<ready>is now ordinary data"


def test_prefix_frame_enforces_exact_prefix_limit_and_fails_closed() -> None:
    frame = rsync_transport.PrefixFrame(b"MARK", max_prefix_bytes=4)
    assert frame.feed(b"1234MARKdata") == b"data"

    overflow = rsync_transport.PrefixFrame(b"MARK", max_prefix_bytes=4)
    with pytest.raises(rsync_transport.FrameError, match="exceeded"):
        overflow.feed(b"12345")
    assert overflow.ready is False

    marker_overflow = rsync_transport.PrefixFrame(b"MARK", max_prefix_bytes=4)
    with pytest.raises(rsync_transport.FrameError, match="exceeded"):
        marker_overflow.feed(b"12345MARKapplication data")
    assert marker_overflow.captured_prefix == b"12345"
    assert b"MARK" not in marker_overflow.captured_prefix
    assert b"application data" not in marker_overflow.captured_prefix


def test_remote_command_does_not_embed_complete_markers_and_preserves_hostile_argv() -> None:
    nonce = "0123456789abcdef0123456789abcdef"
    stdout_marker, stderr_marker = rsync_transport.markers(nonce)
    remote_argv = [
        sys.executable,
        "-c",
        "import sys;sys.stdout.buffer.write(b'OUT\\x00');sys.stderr.buffer.write(b'ERR\\xff')",
        "space and ' quote",
    ]
    command = rsync_transport.build_remote_command(remote_argv, nonce)

    encoded = command.encode("utf-8")
    assert stdout_marker not in encoded
    assert stderr_marker not in encoded
    proc = subprocess.run(["/bin/sh", "-c", command], capture_output=True, check=False)
    assert proc.returncode == 0
    assert proc.stdout == stdout_marker + b"OUT\x00"
    assert proc.stderr == stderr_marker + b"ERR\xff"


@pytest.mark.skipif(shutil.which("csh") is None, reason="csh is not installed")
def test_logical_remote_argv_survives_csh_history_expansion() -> None:
    nonce = "0123456789abcdef0123456789abcdef"
    stdout_marker, stderr_marker = rsync_transport.markers(nonce)
    command = rsync_transport.build_remote_command(
        [sys.executable, "-c", "import sys;print(sys.argv[1])", "bang! space ' quote"],
        nonce,
    )

    proc = subprocess.run(["csh", "-f", "-c", command], capture_output=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == stdout_marker + b"bang! space ' quote\n"
    assert proc.stderr == stderr_marker


def test_transport_command_is_self_contained_and_has_no_checkout_path() -> None:
    command = rsync_transport.build_transport_command(ssh_executable="/usr/bin/ssh")
    argv = shlex.split(command)

    assert Path(argv[0]).name.startswith("python")
    assert "-c" in argv
    source = argv[argv.index("-c") + 1]
    assert "ucl_machine_tools" not in source
    assert "/Users/" not in source
    assert "ucl_rsync_framed_ssh" in source


@pytest.mark.skipif(shutil.which("csh") is None, reason="csh is not installed")
def test_transport_command_survives_csh_login_shell_parsing() -> None:
    command = rsync_transport.build_transport_command(ssh_executable="/definitely/missing/ucl-test-ssh")
    proc = subprocess.run(
        ["csh", "-f", "-c", command + " fake-host true"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Event not found" not in proc.stderr
    assert "could not start SSH transport" in proc.stderr


def test_transport_argv_filters_both_startup_streams_and_preserves_command_bytes() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        [
            sys.executable,
            "-c",
            "import sys;sys.stdout.buffer.write(b'OUT\\x00\\xff');sys.stderr.buffer.write(b'ERR\\x00\\xff')",
        ],
        ssh_executable=str(FAKE_SSH),
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env=_transport_env(
            stdout_prefix=b"missing /tmp/env\n\x00startup",
            stderr_prefix=b"VBoxManage startup\n\xff",
        ),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b"OUT\x00\xff"
    assert proc.stderr == b"ERR\x00\xff"


def test_command_output_that_looks_like_login_noise_is_preserved_after_handshake() -> None:
    command_output = "\n".join(
        [
            "VBoxManage: Failed to create the VirtualBox object",
            "Document is empty",
            "/home/user/.config/VirtualBox/VirtualBox.xml, line 1",
            "NS_ERROR_FAILURE",
            "",
        ]
    )
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        [sys.executable, "-c", f"import sys;sys.stderr.write({command_output!r})"],
        ssh_executable=str(FAKE_SSH),
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env=_transport_env(stderr_prefix=b"VBoxManage startup noise\n"),
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stderr == command_output.encode()


def test_missing_ssh_executable_returns_a_concise_transport_error() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable="/definitely/missing/ucl-test-ssh",
    )
    proc = subprocess.run(argv, capture_output=True, check=False)

    assert proc.returncode != 0
    assert b"could not start ssh transport" in proc.stderr.lower()
    assert b"traceback" not in proc.stderr.lower()


def test_transport_fails_cleanly_when_handshake_never_arrives() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable=str(FAKE_SSH),
        handshake_timeout_seconds=0.2,
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env=_transport_env(stdout_prefix=b"bad startup", stderr_prefix=b"useful startup error", mode="exit"),
        check=False,
    )

    assert proc.returncode != 0
    assert proc.stdout == b""
    assert b"handshake" in proc.stderr.lower()
    assert b"useful startup error" in proc.stderr


def test_transport_failure_hides_virtualbox_startup_block_but_keeps_real_diagnostic() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable=str(FAKE_SSH),
        handshake_timeout_seconds=0.2,
    )
    virtualbox = b"\n".join(
        [
            b"VBoxManage: Failed to create the VirtualBox object",
            b"Document is empty",
            b"/home/user/.config/VirtualBox/VirtualBox.xml, line 1",
            b"NS_ERROR_FAILURE",
            b"ssh: actual transport failure",
            b"",
        ]
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env=_transport_env(stderr_prefix=virtualbox, mode="exit"),
        check=False,
    )

    assert proc.returncode != 0
    assert b"VBoxManage" not in proc.stderr
    assert b"VirtualBox.xml" not in proc.stderr
    assert b"Document is empty" not in proc.stderr
    assert b"NS_ERROR_FAILURE" not in proc.stderr
    assert b"ssh: actual transport failure" in proc.stderr


def test_transport_reports_late_stderr_after_stdout_closes_before_handshake() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable=str(FAKE_SSH),
        handshake_timeout_seconds=1,
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env={
            **_transport_env(mode="stdout_eof_then_error"),
            "FAKE_SSH_LATE_STDERR_B64": _encoded(b"ssh: source host unavailable\n"),
        },
        check=False,
    )

    assert proc.returncode != 0
    assert proc.stdout == b""
    assert b"source host unavailable" in proc.stderr


def test_signal_forwarding_tolerates_child_exit_race(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExitedProcess:
        pid = 12345

        def poll(self) -> None:
            return None

    def vanished_group(pid: int, signum: int) -> None:
        del pid, signum
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", vanished_group)
    rsync_transport._send_signal_if_running(ExitedProcess(), signal.SIGTERM)  # type: ignore[arg-type]


def test_transport_forwards_termination_to_ssh_process_group() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        [sys.executable, "-c", "import time;time.sleep(10)"],
        ssh_executable=str(FAKE_SSH),
    )
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_transport_env(),
    )
    try:
        time.sleep(0.2)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=3)
    except BaseException:
        process.kill()
        process.wait()
        raise

    assert process.returncode == 128 + signal.SIGTERM
    assert stdout == b""
    assert stderr == b""


def test_transport_times_out_before_handshake() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable=str(FAKE_SSH),
        handshake_timeout_seconds=0.1,
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env={**_transport_env(mode="hang"), "FAKE_SSH_HANG_SECONDS": "2"},
        timeout=2,
        check=False,
    )

    assert proc.returncode != 0
    assert b"timed out" in proc.stderr.lower()


def test_transport_rejects_oversized_startup_output() -> None:
    argv = rsync_transport.build_transport_argv(
        "fake-host",
        ["true"],
        ssh_executable=str(FAKE_SSH),
        max_prefix_bytes=8,
    )
    proc = subprocess.run(
        argv,
        capture_output=True,
        env=_transport_env(stdout_prefix=b"123456789", mode="exit"),
        check=False,
    )

    assert proc.returncode != 0
    assert proc.stdout == b""
    assert b"exceeded" in proc.stderr.lower()


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
@pytest.mark.parametrize("direction", ["push", "pull"])
def test_real_rsync_survives_arbitrary_startup_noise(direction: str, tmp_path: Path) -> None:
    source = tmp_path / "source tree"
    destination = tmp_path / "destination tree"
    source.mkdir()
    destination.mkdir()
    relative = Path("nested dir") / "quote ' and space.bin"
    payload = bytes(range(256)) * 64 + b"\x00\xffend"
    source_file = source / relative
    source_file.parent.mkdir()
    source_file.write_bytes(payload)
    transport = rsync_transport.build_transport_command(ssh_executable=str(FAKE_SSH))
    if direction == "push":
        endpoints = [f"{source}/", f"fake-host:{destination}/"]
    else:
        endpoints = [f"fake-host:{source}/", f"{destination}/"]

    proc = subprocess.run(
        ["rsync", "-a", "-e", transport, *endpoints],
        capture_output=True,
        env=_transport_env(
            stdout_prefix=b"Cream login says missing /tmp/ucl-machine-tools/fpt/bin/env\n\x00\xff",
            stderr_prefix=b"VBoxManage: startup warning\n\x00\xff",
        ),
        check=False,
    )

    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    assert (destination / relative).read_bytes() == payload
    assert b"Cream login" not in proc.stdout + proc.stderr
    assert b"VBoxManage" not in proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_copy_builder_forces_safe_remote_arguments_even_when_environment_requests_old_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination; touch SHOULD_NOT_EXIST; #"
    source.mkdir()
    source_file = source / "payload with spaces.txt"
    source_file.write_text("safe", encoding="utf-8")
    transport = rsync_transport.build_transport_command(ssh_executable=str(FAKE_SSH))
    monkeypatch.setattr(copy_tools, "RSYNC_SSH", transport)
    argv = copy_tools.build_rsync_argv(
        copy_tools.Endpoint(str(source) + "/", None, str(source) + "/"),
        copy_tools.Endpoint(f"fake-host:{destination}/", "fake-host", str(destination) + "/"),
    )

    proc = subprocess.run(
        argv,
        capture_output=True,
        env={**_transport_env(), "RSYNC_OLD_ARGS": "1", "RSYNC_PROTECT_ARGS": "0"},
        cwd=tmp_path,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    assert (destination / source_file.name).read_text(encoding="utf-8") == "safe"
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
