#!/usr/bin/env python3
"""Minimal SSH stand-in used by rsync transport integration tests."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time


def _decode(name: str) -> bytes:
    value = os.environ.get(name, "")
    return base64.b64decode(value) if value else b""


def _remote_command(argv: list[str]) -> str:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-o", "-p", "-l", "-i", "-F", "-J"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        index += 1  # host
        break
    if index >= len(argv):
        raise SystemExit("fake ssh did not receive a remote command")
    return " ".join(argv[index:])


def main() -> int:
    os.write(1, _decode("FAKE_SSH_STDOUT_PREFIX_B64"))
    os.write(2, _decode("FAKE_SSH_STDERR_PREFIX_B64"))
    mode = os.environ.get("FAKE_SSH_MODE", "run")
    if mode == "exit":
        return int(os.environ.get("FAKE_SSH_EXIT_CODE", "255"))
    if mode == "hang":
        time.sleep(float(os.environ.get("FAKE_SSH_HANG_SECONDS", "5")))
        return 0
    if mode == "stdout_eof_then_error":
        os.close(1)
        time.sleep(float(os.environ.get("FAKE_SSH_DELAY_SECONDS", "0.05")))
        os.write(2, _decode("FAKE_SSH_LATE_STDERR_B64"))
        return int(os.environ.get("FAKE_SSH_EXIT_CODE", "255"))
    if mode != "run":
        raise SystemExit(f"unsupported FAKE_SSH_MODE: {mode}")
    return subprocess.run(
        ["/bin/sh", "-c", _remote_command(sys.argv[1:])],
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        stderr=sys.stderr.buffer,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
