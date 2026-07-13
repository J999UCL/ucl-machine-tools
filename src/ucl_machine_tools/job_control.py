"""Structured tmux job identity and verified remote stop helpers."""

from __future__ import annotations

import json
import math
from typing import Any


SCHEMA_VERSION = 1
IDENTITY_SENTINEL_BEGIN = "UCL_JOB_IDENTITY_JSON_BEGIN"
IDENTITY_SENTINEL_END = "UCL_JOB_IDENTITY_JSON_END"
LAUNCH_SENTINEL_BEGIN = "UCL_JOB_LAUNCH_JSON_BEGIN"
LAUNCH_SENTINEL_END = "UCL_JOB_LAUNCH_JSON_END"
STOP_SENTINEL_BEGIN = "UCL_JOB_STOP_JSON_BEGIN"
STOP_SENTINEL_END = "UCL_JOB_STOP_JSON_END"
IDENTITY_KEYS = (
    "boot_id",
    "tmux_socket_path",
    "tmux_server_pid",
    "pane_id",
    "window_id",
    "pane_pid",
    "pane_start_ticks",
    "pane_session_id",
)
STOP_STATUSES = {
    "already_stopped",
    "cleanup_failed",
    "helper_error",
    "identity_mismatch",
    "identity_unverified",
    "legacy_unverified",
    "process_snapshot_failed",
    "still_running",
    "stopped",
}


def _extract_sentinel(stdout: str, begin: str, end: str, *, label: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    begin_indices = [index for index, line in enumerate(lines) if line.strip() == begin]
    if len(begin_indices) != 1:
        raise ValueError(f"{label} sentinel must appear exactly once")
    begin_index = begin_indices[0]
    end_indices = [index for index in range(begin_index + 1, len(lines)) if lines[index].strip() == end]
    if len(end_indices) != 1:
        raise ValueError(f"{label} sentinel end must appear exactly once")
    payload_lines = lines[begin_index + 1 : end_indices[0]]
    if len(payload_lines) != 1:
        raise ValueError(f"{label} sentinel payload must be one JSON line")
    try:
        payload = json.loads(payload_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} sentinel payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} sentinel payload must be an object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema: {payload.get('schema_version')!r}")
    if type(payload.get("ok")) is not bool:
        raise ValueError(f"{label} payload field 'ok' must be boolean")
    return payload


def _validate_probe_payload(payload: dict[str, Any], *, label: str) -> dict[str, Any]:
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{label} payload field 'identity' must be an object")
    if "exists" in identity and type(identity["exists"]) is not bool:
        raise ValueError(f"{label} identity field 'exists' must be boolean")
    if "pane_dead" in identity and type(identity["pane_dead"]) is not bool:
        raise ValueError(f"{label} identity field 'pane_dead' must be boolean")
    if "terminal_at_capture" in identity and type(identity["terminal_at_capture"]) is not bool:
        raise ValueError(f"{label} identity field 'terminal_at_capture' must be boolean")
    if "pane_dead_status" in identity and identity["pane_dead_status"] is not None:
        if type(identity["pane_dead_status"]) is not int:
            raise ValueError(f"{label} identity field 'pane_dead_status' must be an integer or null")
    if identity.get("exists") is True:
        if "pane_dead" not in identity:
            raise ValueError(f"live {label} identity must include pane_dead")
        if "pane_dead_status" not in identity:
            raise ValueError(f"live {label} identity must include pane_dead_status")
    if payload["ok"] and type(identity.get("exists")) is not bool:
        raise ValueError(f"successful {label} payload must include identity.exists")
    if not isinstance(payload.get("error", ""), str):
        raise ValueError(f"{label} payload field 'error' must be a string")
    return payload


def parse_identity_stdout(stdout: str) -> dict[str, Any]:
    payload = _extract_sentinel(stdout, IDENTITY_SENTINEL_BEGIN, IDENTITY_SENTINEL_END, label="job identity")
    payload = _validate_probe_payload(payload, label="job identity")
    identity = payload["identity"]
    if payload["ok"] and identity.get("exists") and not has_strong_identity(identity, allow_dead=True):
        raise ValueError("successful job identity payload contains an incomplete live identity")
    return payload


def parse_launch_stdout(stdout: str) -> dict[str, Any]:
    payload = _extract_sentinel(stdout, LAUNCH_SENTINEL_BEGIN, LAUNCH_SENTINEL_END, label="job launch")
    payload = _validate_probe_payload(payload, label="job launch")
    if payload["ok"] and not (
        has_strong_identity(payload["identity"]) or has_terminal_identity(payload["identity"])
    ):
        raise ValueError("successful job launch payload must contain a strong identity")
    if not payload["ok"] and not payload.get("error"):
        raise ValueError("failed job launch payload must include an error")
    return payload


def parse_stop_stdout(stdout: str) -> dict[str, Any]:
    payload = _extract_sentinel(stdout, STOP_SENTINEL_BEGIN, STOP_SENTINEL_END, label="job stop")
    if payload.get("status") not in STOP_STATUSES:
        raise ValueError(f"job stop payload has invalid status: {payload.get('status')!r}")
    for key in ("expected_identity", "current_identity"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"job stop payload field {key!r} must be an object")
    for key in ("signal_errors", "survivors"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"job stop payload field {key!r} must be a list")
    if payload.get("signal") not in {"TERM", "KILL"}:
        raise ValueError("job stop payload field 'signal' must be TERM or KILL")
    for key in ("target", "cleanup"):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"job stop payload field {key!r} must be a string")
    if not isinstance(payload.get("cleanup_error", ""), str):
        raise ValueError("job stop payload field 'cleanup_error' must be a string")
    successful = payload["status"] in {"already_stopped", "stopped"}
    if payload["ok"] != successful:
        raise ValueError("job stop payload ok/status fields contradict each other")
    if successful and payload["survivors"]:
        raise ValueError("successful job stop payload must not contain survivors")
    if payload["status"] == "still_running" and not payload["survivors"]:
        raise ValueError("job stop still_running payload must include survivors")
    if payload["status"] == "cleanup_failed" and not payload.get("cleanup_error"):
        raise ValueError("job stop cleanup_failed payload must include cleanup_error")
    if payload["status"] == "helper_error" and not payload.get("error"):
        raise ValueError("job stop helper_error payload must include a non-empty error string")
    return payload


def has_strong_identity(identity: dict[str, Any], *, allow_dead: bool = False) -> bool:
    if not identity.get("exists"):
        return False
    required = IDENTITY_KEYS[:-2] if allow_dead and identity.get("pane_dead") else IDENTITY_KEYS
    if not all(identity.get(key) not in (None, "") for key in required):
        return False
    string_keys = {"boot_id", "tmux_socket_path", "pane_id", "window_id"}
    integer_keys = {"tmux_server_pid", "pane_pid", "pane_start_ticks", "pane_session_id"}
    return all(isinstance(identity[key], str) for key in string_keys & set(required)) and all(
        type(identity[key]) is int for key in integer_keys & set(required)
    )


def has_terminal_identity(identity: dict[str, Any]) -> bool:
    if (
        identity.get("exists") is not False
        or identity.get("terminal_at_capture") is not True
        or identity.get("pane_dead") is not True
    ):
        return False
    string_keys = ("boot_id", "tmux_socket_path", "pane_id", "window_id")
    integer_keys = ("tmux_server_pid", "pane_pid", "pane_session_id")
    return all(isinstance(identity.get(key), str) and identity[key] for key in string_keys) and all(
        type(identity.get(key)) is int for key in integer_keys
    )


def identity_matches(expected: dict[str, Any], current: dict[str, Any]) -> bool:
    if not has_strong_identity(expected) or not has_strong_identity(current, allow_dead=True):
        return False
    keys = IDENTITY_KEYS[:-2] if current.get("pane_dead") and current.get("pane_start_ticks") is None else IDENTITY_KEYS
    return all(expected.get(key) == current.get(key) for key in keys)


def classify_identity(expected: dict[str, Any], current: dict[str, Any]) -> str:
    if expected.get("pending_launch"):
        return "launch_unknown"
    if has_terminal_identity(expected):
        return "exited_or_missing" if not current.get("exists") else "identity_mismatch"
    if not current.get("exists"):
        return "exited_or_missing"
    if not expected:
        return "legacy_unverified"
    if not expected.get("exists"):
        return "identity_mismatch"
    if not has_strong_identity(expected):
        return "identity_unverified"
    if not identity_matches(expected, current):
        return "identity_mismatch"
    if current.get("pane_dead"):
        return "exited"
    return "running"


def _remote_common_source() -> str:
    return r'''
ABSENT_TMUX_ERRORS = (
    "no server running",
    "can't find session",
    "can't find window",
    "can't find pane",
    "no current target",
)


def boot_id():
    value = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("Linux boot id is empty")
    return value


def proc_stat(pid):
    try:
        text = pathlib.Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        rest = text[text.rfind(")") + 2:].split()
        return {
            "pid": int(pid),
            "state": rest[0],
            "ppid": int(rest[1]),
            "session_id": int(rest[3]),
            "start_ticks": int(rest[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def process_exited(pid):
    if not hasattr(os, "pidfd_open"):
        raise RuntimeError("pidfd inspection is unavailable")
    try:
        pidfd = os.pidfd_open(int(pid))
    except ProcessLookupError:
        return True
    try:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        return bool(poller.poll(0))
    finally:
        os.close(pidfd)


def tmux_target_missing(proc):
    detail = f"{proc.stderr}\n{proc.stdout}".lower()
    if any(marker in detail for marker in ABSENT_TMUX_ERRORS):
        return True
    return "error connecting to" in detail and (
        "no such file or directory" in detail or "connection refused" in detail
    )


def query_target(session, window, pane_id=None, socket_path=None):
    target = pane_id or f"{session}:{window}"
    fmt = "#{pane_id}\t#{window_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pid}\t#{socket_path}"
    tmux = ["tmux"]
    if socket_path:
        tmux.extend(["-S", socket_path])
    proc = subprocess.run(
        [*tmux, "display-message", "-p", "-t", target, fmt],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if tmux_target_missing(proc):
            return {"exists": False, "session": session, "window": window}
        detail = proc.stderr.strip() or proc.stdout.strip() or f"tmux exited {proc.returncode}"
        raise RuntimeError(f"tmux identity query failed: {detail}")
    if not proc.stdout.strip():
        return {"exists": False, "session": session, "window": window}
    fields = proc.stdout.rstrip("\n").split("\t")
    if len(fields) == 7 and not any(fields[:5]):
        return {"exists": False, "session": session, "window": window}
    if len(fields) != 7 or not fields[2].isdigit() or not fields[5].isdigit() or not fields[6]:
        raise RuntimeError("tmux returned malformed pane identity")
    pid = int(fields[2])
    stat = proc_stat(pid)
    return {
        "exists": True,
        "session": session,
        "window": window,
        "boot_id": boot_id(),
        "tmux_socket_path": fields[6],
        "tmux_server_pid": int(fields[5]),
        "pane_id": fields[0],
        "window_id": fields[1],
        "pane_pid": pid,
        "pane_start_ticks": stat["start_ticks"] if stat else None,
        "pane_session_id": stat["session_id"] if stat else None,
        "pane_dead": fields[3] == "1",
        "pane_dead_status": int(fields[4]) if fields[4].lstrip("-").isdigit() else None,
    }


def strong_identity(identity):
    keys = (
        "boot_id", "tmux_socket_path", "tmux_server_pid", "pane_id", "window_id",
        "pane_pid", "pane_start_ticks", "pane_session_id",
    )
    return bool(identity.get("exists")) and all(identity.get(key) not in (None, "") for key in keys)


def same_identity(expected, current):
    if not strong_identity(expected) or not current.get("exists"):
        return False
    keys = ("boot_id", "tmux_socket_path", "tmux_server_pid", "pane_id", "window_id", "pane_pid")
    if not all(expected.get(key) == current.get(key) for key in keys):
        return False
    if current.get("pane_dead"):
        return True
    return (
        expected.get("pane_start_ticks") == current.get("pane_start_ticks")
        and expected.get("pane_session_id") == current.get("pane_session_id")
    )
'''


def build_identity_source(
    session: str,
    window: str,
    expected_identity: dict[str, Any] | None = None,
) -> str:
    params = json.dumps(
        {
            "session": session,
            "window": window,
            "pane_id": (expected_identity or {}).get("pane_id"),
            "socket_path": (expected_identity or {}).get("tmux_socket_path"),
        },
        sort_keys=True,
    )
    return f'''#!/usr/bin/env python3
import json
import os
import pathlib
import select
import subprocess

PARAMS = json.loads({params!r})
BEGIN = {IDENTITY_SENTINEL_BEGIN!r}
END = {IDENTITY_SENTINEL_END!r}
SCHEMA_VERSION = {SCHEMA_VERSION}
{_remote_common_source()}

try:
    identity = query_target(
        PARAMS["session"], PARAMS["window"], PARAMS.get("pane_id"), PARAMS.get("socket_path")
    )
    payload = {{"schema_version": SCHEMA_VERSION, "ok": True, "identity": identity, "error": ""}}
except Exception as exc:
    payload = {{
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "identity": {{"exists": False, "session": PARAMS["session"], "window": PARAMS["window"]}},
        "error": f"{{type(exc).__name__}}: {{exc}}",
    }}
print(BEGIN)
print(json.dumps(payload, sort_keys=True))
print(END)
'''


def build_launch_source(
    mode: str,
    session: str,
    window: str,
    launcher_argv: list[str],
) -> str:
    if mode not in {"new-session", "new-window"}:
        raise ValueError(f"unsupported tmux launch mode: {mode!r}")
    if not launcher_argv:
        raise ValueError("launcher_argv must not be empty")
    params = json.dumps(
        {"mode": mode, "session": session, "window": window, "launcher_argv": launcher_argv},
        sort_keys=True,
    )
    return f'''#!/usr/bin/env python3
import json
import os
import pathlib
import select
import subprocess

PARAMS = json.loads({params!r})
BEGIN = {LAUNCH_SENTINEL_BEGIN!r}
END = {LAUNCH_SENTINEL_END!r}
SCHEMA_VERSION = {SCHEMA_VERSION}
{_remote_common_source()}

fmt = "#{{session_name}}\\t#{{window_id}}\\t#{{window_name}}\\t#{{pane_id}}\\t#{{pane_pid}}\\t#{{pid}}\\t#{{socket_path}}"
if PARAMS["mode"] == "new-session":
    argv = [
        "tmux", "new-session", "-d", "-P", "-F", fmt,
        "-s", PARAMS["session"], "-n", PARAMS["window"],
        *PARAMS["launcher_argv"],
    ]
else:
    argv = [
        "tmux", "new-window", "-d", "-P", "-F", fmt,
        "-t", PARAMS["session"], "-n", PARAMS["window"],
        *PARAMS["launcher_argv"],
    ]
try:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        payload = {{
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "identity": {{}},
            "error": proc.stderr.strip() or proc.stdout.strip() or f"tmux launch exited {{proc.returncode}}",
        }}
    else:
        fields = proc.stdout.rstrip("\\n").split("\\t")
        if len(fields) != 7 or not fields[4].isdigit() or not fields[5].isdigit() or not fields[6]:
            raise RuntimeError("tmux returned malformed launch identity")
        pane_pid = int(fields[4])
        stat = proc_stat(pane_pid)
        bootstrap = {{
            "exists": True,
            "session": fields[0],
            "window": fields[2],
            "boot_id": boot_id(),
            "tmux_socket_path": fields[6],
            "tmux_server_pid": int(fields[5]),
            "window_id": fields[1],
            "pane_id": fields[3],
            "pane_pid": pane_pid,
            "pane_start_ticks": stat["start_ticks"] if stat else None,
            "pane_session_id": stat["session_id"] if stat else pane_pid,
            "pane_dead": False,
            "pane_dead_status": None,
        }}
        try:
            current = query_target(fields[0], fields[2], fields[3], fields[6])
        except Exception as query_exc:
            current = {{**bootstrap, "capture_error": f"{{type(query_exc).__name__}}: {{query_exc}}"}}
        if (
            current.get("exists")
            and not current.get("pane_dead")
            and current.get("pane_start_ticks") is not None
            and current.get("pane_session_id") is not None
        ):
            identity = current
        elif current.get("pane_dead") or not current.get("exists") or process_exited(pane_pid):
            identity = {{
                **bootstrap,
                "exists": False,
                "pane_dead": True,
                "terminal_at_capture": True,
            }}
        else:
            identity = {{**bootstrap, "capture_error": "live pane process identity is unavailable"}}
        payload = {{"schema_version": SCHEMA_VERSION, "ok": True, "identity": identity, "error": ""}}
except Exception as exc:
    payload = {{
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "identity": {{}},
        "error": f"{{type(exc).__name__}}: {{exc}}",
    }}
print(BEGIN)
print(json.dumps(payload, sort_keys=True))
print(END)
'''


def build_stop_source(
    session: str,
    window: str,
    expected_identity: dict[str, Any],
    signal: str,
    grace_seconds: float,
) -> str:
    if signal not in {"TERM", "KILL"}:
        raise ValueError(f"unsupported signal: {signal!r}")
    if not math.isfinite(grace_seconds) or grace_seconds < 0:
        raise ValueError("grace_seconds must be finite and non-negative")
    params = json.dumps(
        {
            "session": session,
            "window": window,
            "expected_identity": expected_identity,
            "signal": signal,
            "grace_seconds": float(grace_seconds),
        },
        sort_keys=True,
    )
    return f'''#!/usr/bin/env python3
import json
import os
import pathlib
import select
import signal as signal_mod
import subprocess
import time

PARAMS = json.loads({params!r})
BEGIN = {STOP_SENTINEL_BEGIN!r}
END = {STOP_SENTINEL_END!r}
SCHEMA_VERSION = {SCHEMA_VERSION}
{_remote_common_source()}


def pidfd_alive(process):
    poller = select.poll()
    poller.register(process["pidfd"], select.POLLIN)
    return not poller.poll(0)


def process_for_output(process):
    return {{
        "pid": process["pid"],
        "start_ticks": process["start_ticks"],
        "session_id": process["session_id"],
    }}


def signal_exact(process, requested_signal, errors):
    if not pidfd_alive(process):
        return True
    try:
        signal_mod.pidfd_send_signal(process["pidfd"], requested_signal)
        return True
    except ProcessLookupError:
        return True
    except Exception as exc:
        errors.append({{"pid": process["pid"], "error": f"{{type(exc).__name__}}: {{exc}}"}})
        return False


def open_session_member(stat, session_id):
    if stat["pid"] == os.getpid() or stat["session_id"] != session_id:
        return None
    try:
        pidfd = os.pidfd_open(stat["pid"])
    except ProcessLookupError:
        return None
    verified = proc_stat(stat["pid"])
    if (
        verified is None
        or verified["start_ticks"] != stat["start_ticks"]
        or verified["session_id"] != session_id
    ):
        os.close(pidfd)
        return None
    return {{
        "pid": verified["pid"],
        "pidfd": pidfd,
        "start_ticks": verified["start_ticks"],
        "session_id": verified["session_id"],
    }}


def session_stats(session_id):
    found = []
    for entry in pathlib.Path("/proc").iterdir():
        if entry.name.isdigit():
            stat = proc_stat(int(entry.name))
            if stat is not None and stat["session_id"] == session_id:
                found.append(stat)
    return found


def capture_session(session_id, tracked):
    added = []
    for stat in session_stats(session_id):
        key = (stat["pid"], stat["start_ticks"])
        if key in tracked:
            continue
        process = open_session_member(stat, session_id)
        if process is None:
            continue
        tracked[key] = process
        added.append(process)
    return added


def capture_owned_session(session_id, root_pid, root_start_ticks):
    if not hasattr(os, "pidfd_open") or not hasattr(signal_mod, "pidfd_send_signal"):
        raise RuntimeError("pidfd signaling is unavailable")
    tracked = {{}}
    errors = []
    stable_rounds = 0
    try:
        root_stat = proc_stat(root_pid)
        if (
            root_stat is None
            or root_stat["start_ticks"] != root_start_ticks
            or root_stat["session_id"] != session_id
        ):
            return None, errors
        root = open_session_member(root_stat, session_id)
        if root is None or not pidfd_alive(root):
            if root is not None:
                os.close(root["pidfd"])
            return None, errors
        tracked[(root_pid, root_start_ticks)] = root
        for _ in range(12):
            if not pidfd_alive(root):
                close_pidfds(tracked)
                return None, errors
            added = capture_session(session_id, tracked)
            stable_rounds = stable_rounds + 1 if not added else 0
            if stable_rounds >= 2:
                return tracked, errors
            time.sleep(0.01)
    except Exception:
        close_pidfds(tracked)
        raise
    close_pidfds(tracked)
    errors.append({{"pid": root_pid, "error": "process session did not stabilize during capture"}})
    return None, errors


def close_pidfds(tracked):
    for process in tracked.values():
        try:
            os.close(process["pidfd"])
        except OSError:
            pass


def signal_and_verify_session(tracked, session_id, root_pid, requested_signal, grace_seconds, errors):
    signaled = set()
    deadline = time.monotonic() + grace_seconds
    empty_rounds = 0
    while True:
        known_alive = [process for process in tracked.values() if pidfd_alive(process)]
        if known_alive:
            capture_session(session_id, tracked)
        ordered = sorted(
            (process for process in tracked.values() if pidfd_alive(process)),
            key=lambda process: process["pid"] == root_pid,
        )
        for process in ordered:
            key = (process["pid"], process["start_ticks"])
            if key not in signaled:
                signal_exact(process, requested_signal, errors)
                signaled.add(key)

        remaining = [process for process in tracked.values() if pidfd_alive(process)]
        if not remaining:
            residual = session_stats(session_id)
            if residual:
                return [
                    {{"pid": stat["pid"], "start_ticks": stat["start_ticks"], "session_id": stat["session_id"]}}
                    for stat in residual
                ]
            empty_rounds += 1
            if empty_rounds >= 2:
                return []
            time.sleep(0.01)
            continue
        else:
            empty_rounds = 0
        if time.monotonic() >= deadline:
            return [process_for_output(process) for process in remaining]
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def stop_job():
    session = PARAMS["session"]
    window = PARAMS["window"]
    expected = PARAMS["expected_identity"]
    target = expected.get("pane_id") or f"{{session}}:{{window}}"
    current = query_target(
        session, window, expected.get("pane_id"), expected.get("tmux_socket_path")
    )
    base = {{
        "target": target,
        "signal": PARAMS["signal"],
        "expected_identity": expected,
        "current_identity": current,
        "signal_errors": [],
        "survivors": [],
        "cleanup": "not_needed",
        "cleanup_error": "",
    }}
    if not expected:
        return {{**base, "ok": False, "status": "legacy_unverified"}}
    if expected.get("terminal_at_capture"):
        if current.get("exists"):
            return {{**base, "ok": False, "status": "identity_mismatch"}}
        members = session_stats(expected["pane_session_id"])
        if members:
            return {{
                **base,
                "ok": False,
                "status": "still_running",
                "survivors": [
                    {{"pid": stat["pid"], "start_ticks": stat["start_ticks"], "session_id": stat["session_id"]}}
                    for stat in members
                ],
            }}
        return {{**base, "ok": True, "status": "already_stopped"}}
    if not expected.get("exists"):
        return {{**base, "ok": False, "status": "identity_mismatch"}}
    if not strong_identity(expected):
        return {{**base, "ok": False, "status": "identity_unverified"}}
    if not current.get("exists"):
        members = session_stats(expected["pane_session_id"])
        if not members:
            return {{**base, "ok": True, "status": "already_stopped"}}
        root = proc_stat(expected["pane_pid"])
        if root is None or root["start_ticks"] != expected["pane_start_ticks"]:
            return {{
                **base,
                "ok": False,
                "status": "identity_unverified",
                "survivors": [
                    {{"pid": stat["pid"], "start_ticks": stat["start_ticks"], "session_id": stat["session_id"]}}
                    for stat in members
                ],
            }}
        current = {{**expected, "pane_dead": False, "pane_dead_status": None}}
        base["current_identity"] = current
    if not same_identity(expected, current):
        return {{**base, "ok": False, "status": "identity_mismatch"}}

    if current.get("pane_dead"):
        members = session_stats(expected["pane_session_id"])
        if members:
            return {{
                **base,
                "ok": False,
                "status": "still_running",
                "survivors": [
                    {{"pid": stat["pid"], "start_ticks": stat["start_ticks"], "session_id": stat["session_id"]}}
                    for stat in members
                ],
            }}
        return {{**base, "ok": True, "status": "already_stopped", "cleanup": "dead_pane_retained"}}

    tracked = None
    signal_errors = []
    try:
        tracked, signal_errors = capture_owned_session(
            current["pane_session_id"], current["pane_pid"], current["pane_start_ticks"]
        )
        if tracked is None:
            return {{
                **base,
                "ok": False,
                "status": "process_snapshot_failed",
                "signal_errors": signal_errors,
            }}
        requested = signal_mod.SIGKILL if PARAMS["signal"] == "KILL" else signal_mod.SIGTERM
        remaining = signal_and_verify_session(
            tracked,
            current["pane_session_id"],
            current["pane_pid"],
            requested,
            PARAMS["grace_seconds"],
            signal_errors,
        )
        if remaining:
            return {{
                **base,
                "ok": False,
                "status": "still_running",
                "signal_errors": signal_errors,
                "survivors": remaining,
            }}
    except Exception as exc:
        return {{
            **base,
            "ok": False,
            "status": "helper_error",
            "signal_errors": signal_errors,
            "error": f"{{type(exc).__name__}}: {{exc}}",
        }}
    finally:
        if tracked is not None:
            close_pidfds(tracked)

    return {{
        **base,
        "ok": True,
        "status": "stopped",
        "signal_errors": signal_errors,
        "cleanup": "not_needed",
    }}


def emit(payload):
    payload["schema_version"] = SCHEMA_VERSION
    print(BEGIN)
    print(json.dumps(payload, sort_keys=True))
    print(END)


try:
    result = stop_job()
except Exception as exc:
    result = {{
        "ok": False,
        "status": "helper_error",
        "target": PARAMS["expected_identity"].get("pane_id") or f"{{PARAMS['session']}}:{{PARAMS['window']}}",
        "signal": PARAMS["signal"],
        "expected_identity": PARAMS["expected_identity"],
        "current_identity": {{}},
        "signal_errors": [],
        "survivors": [],
        "cleanup": "not_attempted",
        "cleanup_error": "",
        "error": f"{{type(exc).__name__}}: {{exc}}",
    }}
emit(result)
'''
