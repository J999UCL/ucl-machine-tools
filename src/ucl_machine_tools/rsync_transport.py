"""Noise-safe SSH transport for remote commands and rsync's binary protocol."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import select
import selectors
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import BinaryIO, Optional, Sequence


DEFAULT_HANDSHAKE_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PREFIX_BYTES = 256 * 1024
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_REMOTE_BOOTSTRAP = (
    'import base64,json,os,sys;'
    'p=json.loads(base64.b64decode(sys.argv[1]));'
    'n=p["nonce"].encode("ascii");'
    'os.write(1,b"\\x1eUCL_RSYNC_STDOUT_"+n+b"\\x1f");'
    'os.write(2,b"\\x1eUCL_RSYNC_STDERR_"+n+b"\\x1f");'
    'a=p["argv"] if p["mode"]=="argv" else ["/bin/sh","-c"," ".join(p["words"])];'
    'os.execvp(a[0],a)'
)


class FrameError(RuntimeError):
    """Raised when a remote transport handshake cannot be framed safely."""


_VIRTUALBOX_STARTUP_BLOCK_RE = re.compile(
    rb"^VBoxManage: Failed to create the VirtualBox object\r?\n"
    rb"Document is empty\r?\n"
    rb"[^\r\n]*/\.config/VirtualBox/VirtualBox\.xml, line 1\r?\n"
    rb"NS_ERROR_FAILURE(?:\r?\n|$)",
    re.MULTILINE,
)


def strip_virtualbox_startup_noise(data: bytes) -> bytes:
    """Remove only complete known VirtualBox login-hook blocks."""

    return _VIRTUALBOX_STARTUP_BLOCK_RE.sub(b"", data)


class PrefixFrame:
    """Discard a bounded byte prefix through a marker, then pass bytes unchanged."""

    def __init__(self, marker: bytes, max_prefix_bytes: int) -> None:
        if not marker:
            raise ValueError("frame marker must be non-empty")
        if max_prefix_bytes < 0:
            raise ValueError("max_prefix_bytes must be non-negative")
        self.marker = marker
        self.max_prefix_bytes = max_prefix_bytes
        self.ready = False
        self.prefix = b""
        self._prefix_captured = False
        self._buffer = bytearray()

    @property
    def captured_prefix(self) -> bytes:
        return self.prefix if self._prefix_captured else bytes(self._buffer)

    def feed(self, data: bytes) -> bytes:
        if self.ready:
            return data
        self._buffer.extend(data)
        index = self._buffer.find(self.marker)
        if index >= 0:
            self.prefix = bytes(self._buffer[:index])
            self._prefix_captured = True
            payload = bytes(self._buffer[index + len(self.marker) :])
            self._buffer.clear()
            if index > self.max_prefix_bytes:
                raise FrameError(
                    f"startup output exceeded {self.max_prefix_bytes} bytes before the transport handshake"
                )
            self.ready = True
            return payload

        overlap = 0
        maximum = min(len(self._buffer), len(self.marker) - 1)
        for length in range(maximum, 0, -1):
            if self._buffer.endswith(self.marker[:length]):
                overlap = length
                break
        definite_prefix = len(self._buffer) - overlap
        if definite_prefix > self.max_prefix_bytes:
            raise FrameError(
                f"startup output exceeded {self.max_prefix_bytes} bytes before the transport handshake"
            )
        return b""


def markers(nonce: str) -> tuple[bytes, bytes]:
    if not _NONCE_RE.fullmatch(nonce):
        raise ValueError("transport nonce must be 32 lowercase hexadecimal characters")
    token = nonce.encode("ascii")
    return (
        b"\x1eUCL_RSYNC_STDOUT_" + token + b"\x1f",
        b"\x1eUCL_RSYNC_STDERR_" + token + b"\x1f",
    )


def build_remote_command(remote_argv: Sequence[str], nonce: str) -> str:
    if not remote_argv:
        raise ValueError("remote transport command must be non-empty")
    markers(nonce)
    if any("\x00" in part for part in remote_argv):
        raise ValueError("remote transport arguments cannot contain NUL bytes")
    return _build_remote_bootstrap({"nonce": nonce, "mode": "argv", "argv": list(remote_argv)})


def _build_remote_command_from_shell_words(remote_words: Sequence[str], nonce: str) -> str:
    """Frame rsync's already shell-escaped remote command words."""

    markers(nonce)
    if not remote_words:
        raise ValueError("remote transport command must be non-empty")
    if any("\x00" in word for word in remote_words):
        raise ValueError("remote transport arguments cannot contain NUL bytes")
    return _build_remote_bootstrap({"nonce": nonce, "mode": "shell_words", "words": list(remote_words)})


def _build_remote_bootstrap(params: dict[str, object]) -> str:
    payload = base64.b64encode(json.dumps(params, separators=(",", ":")).encode("utf-8")).decode("ascii")
    # These tokens contain no csh history-expansion character. The account's
    # login shell only has to start Python; arbitrary argv stays Base64-encoded.
    if "!" in _REMOTE_BOOTSTRAP or "!" in payload:
        raise AssertionError("remote bootstrap must remain csh-safe")
    return shlex.join(["python3", "-c", _REMOTE_BOOTSTRAP, payload])


def _embedded_source() -> str:
    source = Path(__file__).read_bytes()
    # Standard Base64 deliberately avoids `!`, which csh expands even inside
    # single quotes while parsing an SSH remote command.
    payload = base64.b64encode(zlib.compress(source, level=9)).decode("ascii")
    return (
        "import base64,zlib;"
        f"ucl_rsync_framed_ssh={payload!r};"
        "exec(compile(zlib.decompress(base64.b64decode(ucl_rsync_framed_ssh)),"
        "'<ucl-rsync-transport>','exec'))"
    )


def build_transport_command(
    *,
    ssh_executable: str = "ssh",
    handshake_timeout_seconds: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
    max_prefix_bytes: int = DEFAULT_MAX_PREFIX_BYTES,
    logical_argv: bool = False,
    forward_agent: bool = False,
) -> str:
    argv = [
        "python3",
        "-c",
        _embedded_source(),
        "--ssh",
        ssh_executable,
        "--handshake-timeout",
        str(float(handshake_timeout_seconds)),
        "--max-prefix-bytes",
        str(int(max_prefix_bytes)),
    ]
    if logical_argv:
        argv.append("--logical-argv")
    if forward_agent:
        argv.append("--forward-agent")
    return shlex.join([*argv, "--"])


def build_transport_argv(
    host: str,
    remote_argv: Sequence[str],
    *,
    ssh_executable: str = "ssh",
    handshake_timeout_seconds: float = DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
    max_prefix_bytes: int = DEFAULT_MAX_PREFIX_BYTES,
    forward_agent: bool = False,
) -> list[str]:
    if not host or host.startswith("-") or "\x00" in host:
        raise ValueError(f"unsafe SSH host: {host!r}")
    command = build_transport_command(
        ssh_executable=ssh_executable,
        handshake_timeout_seconds=handshake_timeout_seconds,
        max_prefix_bytes=max_prefix_bytes,
        logical_argv=True,
        forward_agent=forward_agent,
    )
    return [*shlex.split(command), host, *remote_argv]


def _parse_remote_shell_args(args: Sequence[str]) -> tuple[list[str], str, list[str]]:
    ssh_options: list[str] = []
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option == "-l":
            if index + 1 >= len(args):
                raise FrameError("rsync remote shell supplied -l without a user")
            ssh_options.extend((option, args[index + 1]))
            index += 2
            continue
        if option in {"-4", "-6"}:
            ssh_options.append(option)
            index += 1
            continue
        raise FrameError(f"unsupported rsync remote-shell option: {option}")
    if index >= len(args):
        raise FrameError("rsync remote shell did not supply a host")
    host = args[index]
    remote_argv = list(args[index + 1 :])
    if not remote_argv:
        raise FrameError("rsync remote shell did not supply a remote command")
    return ssh_options, host, remote_argv


def _write_all(stream: BinaryIO, data: bytes) -> None:
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        descriptor = None
    if descriptor is not None:
        view = memoryview(data)
        while view:
            try:
                written = os.write(descriptor, view)
            except BlockingIOError:
                select.select([], [descriptor], [])
                continue
            if written <= 0:
                raise BrokenPipeError("transport output stream closed")
            view = view[written:]
        return

    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None:
            written = len(view)
        if written <= 0:
            raise BrokenPipeError("transport output stream closed")
        view = view[written:]
    stream.flush()


def _preview(data: bytes, limit: int = 512) -> str:
    text = strip_virtualbox_startup_noise(data).decode("utf-8", errors="backslashreplace")
    clipped = text[:limit]
    if len(text) > limit:
        clipped += f"... ({len(text) - limit} more characters)"
    return clipped


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _send_signal_if_running(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        if process.poll() is None:
            os.killpg(process.pid, signum)
    except ProcessLookupError:
        # The child exited between poll() and send_signal().
        return


def _establish_frames(
    process: subprocess.Popen[bytes],
    frames: dict[str, PrefixFrame],
    handshake_timeout_seconds: float,
) -> dict[str, bytes]:
    if process.stdout is None or process.stderr is None:
        raise FrameError("SSH transport pipes were not created")
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    pending = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    for name, stream in streams.items():
        selector.register(stream, selectors.EVENT_READ, name)
    deadline = time.monotonic() + handshake_timeout_seconds
    failed_streams: set[str] = set()
    diagnostic_deadline: Optional[float] = None

    def closed_error() -> FrameError:
        names = ", ".join(sorted(failed_streams))
        return FrameError(f"SSH {names} closed before the transport handshake")

    try:
        while not all(frame.ready for frame in frames.values()):
            active_deadline = diagnostic_deadline if diagnostic_deadline is not None else deadline
            timeout = active_deadline - time.monotonic()
            if timeout <= 0:
                if failed_streams:
                    raise closed_error()
                raise FrameError(f"transport handshake timed out after {handshake_timeout_seconds:g} seconds")
            events = selector.select(timeout)
            if not events:
                if failed_streams:
                    raise closed_error()
                raise FrameError(f"transport handshake timed out after {handshake_timeout_seconds:g} seconds")
            for key, _ in events:
                name = key.data
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    if not frames[name].ready:
                        failed_streams.add(name)
                        if diagnostic_deadline is None:
                            diagnostic_deadline = deadline
                    continue
                payload = frames[name].feed(chunk)
                if frames[name].ready:
                    pending[name].extend(payload)
                    selector.unregister(key.fileobj)
            if failed_streams and not selector.get_map():
                raise closed_error()
    finally:
        selector.close()
    return {name: bytes(data) for name, data in pending.items()}


def _pump_stream(
    source: BinaryIO,
    destination: BinaryIO,
    initial: bytes,
    process: subprocess.Popen[bytes],
    errors: list[Exception],
    error_lock: threading.Lock,
) -> None:
    try:
        if initial:
            _write_all(destination, initial)
        while True:
            chunk = os.read(source.fileno(), 64 * 1024)
            if not chunk:
                return
            _write_all(destination, chunk)
    except (BrokenPipeError, OSError) as exc:
        with error_lock:
            first_error = not errors
            errors.append(exc)
        if first_error:
            _stop_child(process)


def _proxy(
    ssh_argv: Sequence[str],
    *,
    stdout_marker: bytes,
    stderr_marker: bytes,
    handshake_timeout_seconds: float,
    max_prefix_bytes: int,
) -> int:
    forwarded_signal_numbers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    forwarded_signals: dict[int, object] = {}
    pending_signals: list[int] = []
    process: Optional[subprocess.Popen[bytes]] = None

    def forward_signal(signum: int, _frame: object) -> None:
        if process is None:
            pending_signals.append(signum)
        else:
            _send_signal_if_running(process, signum)

    try:
        for signum in forwarded_signal_numbers:
            forwarded_signals[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
    except Exception as exc:
        for signum, previous in forwarded_signals.items():
            signal.signal(signum, previous)
        raise FrameError(f"could not install SSH transport signal handlers: {exc}") from exc

    try:
        process = subprocess.Popen(
            list(ssh_argv),
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            start_new_session=True,
        )
    except Exception as exc:
        for signum, previous in forwarded_signals.items():
            signal.signal(signum, previous)
        raise FrameError(f"could not start SSH transport: {exc}") from exc
    for signum in pending_signals:
        _send_signal_if_running(process, signum)

    frames = {
        "stdout": PrefixFrame(stdout_marker, max_prefix_bytes),
        "stderr": PrefixFrame(stderr_marker, max_prefix_bytes),
    }
    outputs: dict[str, BinaryIO] = {
        "stdout": sys.stdout.buffer,
        "stderr": sys.stderr.buffer,
    }
    failure: Optional[Exception] = None
    returncode = 255

    try:
        pending = _establish_frames(process, frames, handshake_timeout_seconds)
        if process.stdout is None or process.stderr is None:
            raise FrameError("SSH transport pipes were not created")
        errors: list[Exception] = []
        error_lock = threading.Lock()
        threads = [
            threading.Thread(
                target=_pump_stream,
                args=(process.stdout, outputs["stdout"], pending["stdout"], process, errors, error_lock),
                name="ucl-rsync-stdout",
            ),
            threading.Thread(
                target=_pump_stream,
                args=(process.stderr, outputs["stderr"], pending["stderr"], process, errors, error_lock),
                name="ucl-rsync-stderr",
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            failure = errors[0]
        else:
            returncode = process.wait()
    except FrameError as exc:
        failure = exc
        _stop_child(process)
    except Exception as exc:  # Keep transport failures concise and reap SSH.
        failure = exc
        _stop_child(process)
    finally:
        for signum, previous in forwarded_signals.items():
            signal.signal(signum, previous)

    if failure is not None:
        details = [f"ucl remote transport failed: {failure}"]
        for name in ("stdout", "stderr"):
            prefix = frames[name].captured_prefix
            if prefix:
                preview = _preview(prefix)
                if preview.strip():
                    details.append(f"remote {name} before handshake: {preview}")
        try:
            _write_all(sys.stderr.buffer, ("\n".join(details) + "\n").encode("utf-8"))
        except BrokenPipeError:
            pass
        return 255
    return 128 + (-returncode) if returncode < 0 else returncode


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    tokens = list(argv)
    try:
        separator = tokens.index("--")
    except ValueError as exc:
        raise FrameError("transport invocation is missing the argument separator") from exc
    parser = argparse.ArgumentParser(prog="ucl-rsync-transport", add_help=False)
    parser.add_argument("--ssh", default="ssh")
    parser.add_argument("--handshake-timeout", type=float, default=DEFAULT_HANDSHAKE_TIMEOUT_SECONDS)
    parser.add_argument("--max-prefix-bytes", type=int, default=DEFAULT_MAX_PREFIX_BYTES)
    parser.add_argument("--logical-argv", action="store_true")
    parser.add_argument("--forward-agent", action="store_true")
    options = parser.parse_args(tokens[:separator])
    if options.handshake_timeout <= 0:
        raise FrameError("handshake timeout must be positive")
    if options.max_prefix_bytes < 0:
        raise FrameError("maximum startup prefix must be non-negative")
    return options, tokens[separator + 1 :]


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        options, remote_shell_args = _parse_args(sys.argv[1:] if argv is None else argv)
        ssh_options, host, remote_argv = _parse_remote_shell_args(remote_shell_args)
        nonce = uuid.uuid4().hex
        stdout_marker, stderr_marker = markers(nonce)
        if options.logical_argv:
            remote_command = build_remote_command(remote_argv, nonce)
        else:
            # Rsync deliberately supplies shell-escaped remote words to its rsh.
            # Preserve that representation so the login shell decodes it once.
            remote_command = _build_remote_command_from_shell_words(remote_argv, nonce)
        ssh_argv = [
            options.ssh,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
            "-o",
            f"ConnectTimeout={max(1, int(options.handshake_timeout))}",
            *(["-A"] if options.forward_agent else []),
            *ssh_options,
            host,
            remote_command,
        ]
        return _proxy(
            ssh_argv,
            stdout_marker=stdout_marker,
            stderr_marker=stderr_marker,
            handshake_timeout_seconds=options.handshake_timeout,
            max_prefix_bytes=options.max_prefix_bytes,
        )
    except FrameError as exc:
        print(f"ucl remote transport: {exc}", file=sys.stderr)
        return 255


if __name__ == "__main__":
    raise SystemExit(main())
