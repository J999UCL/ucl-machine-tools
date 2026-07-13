from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from ucl_machine_tools import job_control


Parser = Callable[[str], dict[str, Any]]


def sentinel(begin: str, end: str, payload: object) -> str:
    return "\n".join(["login noise", begin, json.dumps(payload), end, "logout noise"])


def strong_identity(**overrides: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "exists": True,
        "session": "demo",
        "window": "run",
        "boot_id": "boot-a",
        "tmux_socket_path": "/tmp/tmux-1/default",
        "tmux_server_pid": 101,
        "pane_id": "%2",
        "window_id": "@1",
        "pane_pid": 123,
        "pane_start_ticks": 456,
        "pane_session_id": 123,
        "pane_dead": False,
        "pane_dead_status": None,
    }
    identity.update(overrides)
    return identity


def probe_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": job_control.SCHEMA_VERSION,
        "ok": True,
        "identity": {"exists": False},
        "error": "",
    }
    payload.update(overrides)
    return payload


def stop_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": job_control.SCHEMA_VERSION,
        "ok": True,
        "status": "already_stopped",
        "signal": "TERM",
        "target": "%2",
        "expected_identity": {},
        "current_identity": {"exists": False},
        "signal_errors": [],
        "survivors": [],
        "cleanup": "not_needed",
        "cleanup_error": "",
    }
    payload.update(overrides)
    return payload


def proc_start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(text[text.rfind(")") + 2 :].split()[19])


def linux_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def fake_tmux(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gone = tmp_path / "gone"
    script = bin_dir / "tmux"
    script.write_text(
        """#!/usr/bin/env bash
set -eu
if [ "$1" = "-S" ]; then
  shift 2
fi
printf '%s\\n' "$*" >> "$FAKE_TMUX_LOG"
case "$1" in
  display-message)
    if [ -e "$FAKE_TMUX_GONE" ]; then
      printf "can't find pane\\n" >&2
      exit 1
    fi
    displays=$(grep -c '^display-message' "$FAKE_TMUX_LOG" || true)
    pane_dead=0
    if [ "$displays" -gt 1 ]; then
      pane_dead=1
    fi
    printf '%%%s\\t@%s\\t%s\\t%s\\t\\t%s\\t%s\\n' \
      "$FAKE_PANE_ID" "$FAKE_WINDOW_ID" "$FAKE_PANE_PID" "$pane_dead" "$FAKE_TMUX_SERVER_PID" "$FAKE_TMUX_SOCKET"
    ;;
  kill-pane)
    : > "$FAKE_TMUX_GONE"
    ;;
  *)
    printf 'unexpected tmux command: %s\\n' "$*" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bin_dir, gone


def test_sentinel_parsers_ignore_noise_outside_valid_payloads() -> None:
    identity = probe_payload()
    launch = probe_payload(identity=strong_identity())
    stop = stop_payload()

    assert job_control.parse_identity_stdout(
        sentinel(job_control.IDENTITY_SENTINEL_BEGIN, job_control.IDENTITY_SENTINEL_END, identity)
    ) == identity
    assert job_control.parse_launch_stdout(
        sentinel(job_control.LAUNCH_SENTINEL_BEGIN, job_control.LAUNCH_SENTINEL_END, launch)
    ) == launch
    assert job_control.parse_stop_stdout(
        sentinel(job_control.STOP_SENTINEL_BEGIN, job_control.STOP_SENTINEL_END, stop)
    ) == stop

    with pytest.raises(ValueError, match="exactly once"):
        job_control.parse_stop_stdout("no sentinel")


@pytest.mark.parametrize(
    ("parser", "begin", "end"),
    [
        (
            job_control.parse_identity_stdout,
            job_control.IDENTITY_SENTINEL_BEGIN,
            job_control.IDENTITY_SENTINEL_END,
        ),
        (
            job_control.parse_launch_stdout,
            job_control.LAUNCH_SENTINEL_BEGIN,
            job_control.LAUNCH_SENTINEL_END,
        ),
    ],
)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "1"}, "unsupported"),
        ({"schema_version": True}, "unsupported"),
        ({"schema_version": 999}, "unsupported"),
        ({"ok": "false"}, "must be boolean"),
        ({"ok": 1}, "must be boolean"),
        ({"identity": []}, "identity.*object"),
        ({"identity": {"exists": "yes"}}, "exists.*boolean"),
        ({"identity": {"exists": False, "pane_dead": "false"}}, "pane_dead.*boolean"),
        ({"identity": {}}, "must include identity.exists"),
        ({"error": None}, "error.*string"),
    ],
)
def test_probe_parsers_reject_invalid_schema_and_field_types(
    parser: Parser,
    begin: str,
    end: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parser(sentinel(begin, end, probe_payload(**overrides)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "1"}, "unsupported"),
        ({"schema_version": True}, "unsupported"),
        ({"ok": "false"}, "must be boolean"),
        ({"status": "definitely_stopped"}, "invalid status"),
        ({"signal": "INT"}, "signal.*TERM or KILL"),
        ({"target": None}, "target.*string"),
        ({"expected_identity": []}, "expected_identity.*object"),
        ({"current_identity": None}, "current_identity.*object"),
        ({"signal_errors": {}}, "signal_errors.*list"),
        ({"survivors": {}}, "survivors.*list"),
        ({"cleanup": None}, "cleanup.*string"),
        ({"cleanup_error": []}, "cleanup_error.*string"),
    ],
)
def test_stop_parser_rejects_invalid_schema_status_and_field_types(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        job_control.parse_stop_stdout(
            sentinel(
                job_control.STOP_SENTINEL_BEGIN,
                job_control.STOP_SENTINEL_END,
                stop_payload(**overrides),
            )
        )


@pytest.mark.parametrize(
    ("parser", "begin", "end"),
    [
        (
            job_control.parse_identity_stdout,
            job_control.IDENTITY_SENTINEL_BEGIN,
            job_control.IDENTITY_SENTINEL_END,
        ),
        (
            job_control.parse_launch_stdout,
            job_control.LAUNCH_SENTINEL_BEGIN,
            job_control.LAUNCH_SENTINEL_END,
        ),
        (
            job_control.parse_stop_stdout,
            job_control.STOP_SENTINEL_BEGIN,
            job_control.STOP_SENTINEL_END,
        ),
    ],
)
def test_parsers_reject_non_object_payloads(parser: Parser, begin: str, end: str) -> None:
    with pytest.raises(ValueError, match="must be an object"):
        parser(sentinel(begin, end, []))


def test_launch_parser_requires_a_complete_strong_identity() -> None:
    with pytest.raises(ValueError, match="pane_dead|strong identity"):
        job_control.parse_launch_stdout(
            sentinel(
                job_control.LAUNCH_SENTINEL_BEGIN,
                job_control.LAUNCH_SENTINEL_END,
                probe_payload(identity={"exists": True}),
            )
        )


def test_launch_parser_accepts_a_verified_terminal_at_capture_identity() -> None:
    terminal = strong_identity(
        exists=False,
        pane_dead=True,
        pane_start_ticks=None,
        pane_session_id=123,
        terminal_at_capture=True,
    )
    payload = probe_payload(identity=terminal)

    assert job_control.parse_launch_stdout(
        sentinel(job_control.LAUNCH_SENTINEL_BEGIN, job_control.LAUNCH_SENTINEL_END, payload)
    ) == payload
    assert job_control.classify_identity(terminal, {"exists": False}) == "exited_or_missing"


def test_probe_parser_requires_explicit_live_and_terminal_state_fields() -> None:
    live = strong_identity()
    live.pop("pane_dead")
    with pytest.raises(ValueError, match="must include pane_dead"):
        job_control.parse_identity_stdout(
            sentinel(
                job_control.IDENTITY_SENTINEL_BEGIN,
                job_control.IDENTITY_SENTINEL_END,
                probe_payload(identity=live),
            )
        )

    terminal = strong_identity(
        exists=False,
        pane_dead=False,
        pane_start_ticks=None,
        pane_session_id=123,
        terminal_at_capture=True,
    )
    with pytest.raises(ValueError, match="strong identity"):
        job_control.parse_launch_stdout(
            sentinel(
                job_control.LAUNCH_SENTINEL_BEGIN,
                job_control.LAUNCH_SENTINEL_END,
                probe_payload(identity=terminal),
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        stop_payload(ok=True, status="still_running"),
        stop_payload(ok=False, status="stopped"),
        stop_payload(ok=True, status="stopped", survivors=[{"pid": 4}]),
        stop_payload(ok=False, status="still_running", survivors=[]),
        stop_payload(ok=False, status="cleanup_failed", cleanup_error=""),
        stop_payload(ok=False, status="helper_error", error=""),
    ],
)
def test_stop_parser_rejects_semantically_contradictory_results(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="contradict|survivors|cleanup_error|error string"):
        job_control.parse_stop_stdout(
            sentinel(job_control.STOP_SENTINEL_BEGIN, job_control.STOP_SENTINEL_END, payload)
        )


def test_identity_classification_requires_boot_server_and_exact_recorded_pane() -> None:
    expected = strong_identity()
    current = strong_identity()

    assert job_control.classify_identity(expected, current) == "running"
    assert job_control.classify_identity(
        expected,
        strong_identity(pane_dead=True, pane_start_ticks=None),
    ) == "exited"
    assert job_control.classify_identity(expected, {"exists": False}) == "exited_or_missing"
    assert job_control.classify_identity({"pending_launch": True}, current) == "launch_unknown"
    assert job_control.classify_identity({}, current) == "legacy_unverified"
    assert job_control.classify_identity({"exists": False}, current) == "identity_mismatch"
    assert job_control.classify_identity(strong_identity(boot_id=None), current) == "identity_unverified"
    assert job_control.classify_identity(strong_identity(tmux_server_pid=None), current) == "identity_unverified"
    assert job_control.classify_identity(expected, strong_identity(boot_id="boot-b")) == "identity_mismatch"
    assert job_control.classify_identity(expected, strong_identity(tmux_server_pid=202)) == "identity_mismatch"
    assert job_control.classify_identity(expected, strong_identity(pane_pid=999)) == "identity_mismatch"


def test_generated_helpers_use_strong_identity_without_deleting_tmux_state() -> None:
    expected = strong_identity()
    identity_source = job_control.build_identity_source("demo", "run", expected)
    stop_source = job_control.build_stop_source("demo", "run", expected, "TERM", 5)
    launch_source = job_control.build_launch_source(
        "new-session",
        "demo",
        "run",
        ["bash", "/tmp/demo/.ucl_payload.sh"],
    )

    compile(identity_source, "<identity-helper>", "exec")
    compile(stop_source, "<stop-helper>", "exec")
    compile(launch_source, "<launch-helper>", "exec")

    assert "kill-pane" not in stop_source
    assert "kill-window" not in stop_source
    assert "kill-session" not in stop_source
    assert "os.killpg" not in stop_source
    assert "pidfd_open" in stop_source
    assert "SIGSTOP" not in stop_source
    assert "SIGCONT" not in stop_source
    assert '"boot_id", "tmux_socket_path", "tmux_server_pid", "pane_id"' in stop_source

    assert "#{pane_id}\\t#{window_id}\\t#{pane_pid}" in identity_source
    assert "#{pane_dead_status}\\t#{pid}\\t#{socket_path}" in identity_source
    assert '"boot_id": boot_id()' in launch_source
    assert '"tmux_server_pid": int(fields[5])' in launch_source
    assert '"tmux_socket_path": fields[6]' in launch_source
    assert '"pane_session_id": stat["session_id"] if stat else pane_pid' in launch_source
    assert "process_exited(pane_pid)" in launch_source
    assert '"tmux", "new-session", "-d", "-P", "-F", fmt' in launch_source
    assert '"-s", PARAMS["session"], "-n", PARAMS["window"]' in launch_source


def test_term_helper_never_silently_escalates_to_kill() -> None:
    source = job_control.build_stop_source("demo", "run", strong_identity(), "TERM", 5)

    assert 'requested = signal_mod.SIGKILL if PARAMS["signal"] == "KILL" else signal_mod.SIGTERM' in source
    assert '"status": "still_running"' in source
    assert "signal_exact(process, signal_mod.SIGKILL" not in source
    assert "signal_exact(root, signal_mod.SIGKILL" not in source
    assert "requested = signal_mod.SIGKILL\n" not in source


@pytest.mark.parametrize("grace", [float("nan"), float("inf"), -1.0])
def test_stop_helper_rejects_invalid_grace_periods(grace: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        job_control.build_stop_source("demo", "run", strong_identity(), "TERM", grace)


def test_identity_helper_treats_tmux_blank_target_fields_as_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\nprintf '\\t\\t\\t\\t\\t4242\\t/tmp/tmux-1/default\\n'\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, "-"],
        input=job_control.build_identity_source("missing", "window"),
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        check=True,
    )

    payload = job_control.parse_identity_stdout(proc.stdout)
    assert payload["ok"] is True
    assert payload["identity"] == {"exists": False, "session": "missing", "window": "window"}


def test_identity_helper_treats_a_gone_tmux_socket_as_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'error connecting to /tmp/tmux-1/default (No such file or directory)\\n' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, "-"],
        input=job_control.build_identity_source(
            "missing",
            "window",
            {"tmux_socket_path": "/tmp/tmux-1/default"},
        ),
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        check=True,
    )

    payload = job_control.parse_identity_stdout(proc.stdout)
    assert payload["ok"] is True
    assert payload["identity"] == {"exists": False, "session": "missing", "window": "window"}


@pytest.mark.skipif(sys.platform != "linux", reason="generated stop helper requires Linux /proc")
def test_stop_helper_refuses_a_live_legacy_target(tmp_path: Path) -> None:
    bin_dir, gone = fake_tmux(tmp_path)
    log = tmp_path / "tmux.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_TMUX_LOG": str(log),
        "FAKE_TMUX_GONE": str(gone),
        "FAKE_PANE_ID": "3",
        "FAKE_WINDOW_ID": "4",
        "FAKE_PANE_PID": str(os.getpid()),
        "FAKE_TMUX_SERVER_PID": "101",
        "FAKE_TMUX_SOCKET": "/tmp/tmux-1/default",
    }
    proc = subprocess.run(
        [sys.executable, "-"],
        input=job_control.build_stop_source("demo", "run", {}, "TERM", 0),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    payload = job_control.parse_stop_stdout(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "legacy_unverified"
    tmux_log = log.read_text(encoding="utf-8")
    assert "kill-pane" not in tmux_log
    assert "kill-window" not in tmux_log
    assert "kill-session" not in tmux_log


@pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="process stop verification requires Linux /proc and pidfd signaling",
)
def test_stop_helper_terminates_the_recorded_process_session_without_tmux_cleanup(tmp_path: Path) -> None:
    bin_dir, gone = fake_tmux(tmp_path)
    log = tmp_path / "tmux.log"
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        expected = strong_identity(
            boot_id=linux_boot_id(),
            tmux_server_pid=4242,
            tmux_socket_path="/tmp/tmux-1/default",
            pane_id="%7",
            window_id="@9",
            pane_pid=child.pid,
            pane_start_ticks=proc_start_ticks(child.pid),
            pane_session_id=child.pid,
        )
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_TMUX_LOG": str(log),
            "FAKE_TMUX_GONE": str(gone),
            "FAKE_PANE_ID": "7",
            "FAKE_WINDOW_ID": "9",
            "FAKE_PANE_PID": str(child.pid),
            "FAKE_TMUX_SERVER_PID": "4242",
            "FAKE_TMUX_SOCKET": "/tmp/tmux-1/default",
        }
        proc = subprocess.run(
            [sys.executable, "-"],
            input=job_control.build_stop_source("demo", "run", expected, "TERM", 2),
            capture_output=True,
            text=True,
            env=env,
            timeout=8,
            check=True,
        )
        payload = job_control.parse_stop_stdout(proc.stdout)

        assert payload["ok"] is True
        assert payload["status"] == "stopped"
        assert payload["survivors"] == []
        assert payload["cleanup"] == "not_needed"
        tmux_log = log.read_text(encoding="utf-8")
        assert "kill-pane" not in tmux_log
        assert "kill-window" not in tmux_log
        assert "kill-session" not in tmux_log
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
