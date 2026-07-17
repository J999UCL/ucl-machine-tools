from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from ucl_machine_tools import uv_remote


PROJECT_LOCK_SHA256 = hashlib.sha256(b"version = 1\n").hexdigest()


def make_spec(tmp_path: Path, **overrides: object) -> uv_remote.UvSetupSpec:
    root = tmp_path / "remote root"
    paths = uv_remote.UvRemotePaths(
        source_dir=str(root / "source"),
        environment_dir=str(root / "envs" / "environment"),
        uv_cache_dir=str(root / "cache" / "uv"),
        uv_tool_dir=str(root / "tools" / "uv" / "0.8.14"),
        uv_binary_path=str(root / "tools" / "uv" / "0.8.14" / "uv"),
        python_install_dir=str(root / "tools" / "python"),
        ready_state_path=str(root / "state" / "ready.json"),
        failed_state_path=str(root / "state" / "failed.json"),
        log_path=str(root / "logs" / "setup.log"),
        environment_lock_path=str(root / "locks" / "environment.lock"),
        uv_tool_lock_path=str(root / "locks" / "uv-tool.lock"),
    )
    values: dict[str, object] = {
        "uv_version": "0.8.14",
        "paths": paths,
        "source_sha256": "a" * 64,
        "lock_sha256": PROJECT_LOCK_SHA256,
        "setup_environment_sha256": "c" * 64,
        "python_request": "3.11.5",
        "gpu_id": "0",
        "cuda_visibility_script": "/usr/local/cuda/CUDA_VISIBILITY.csh",
    }
    values.update(overrides)
    return uv_remote.UvSetupSpec(**values)  # type: ignore[arg-type]


def build_payload(tmp_path: Path, **overrides: object) -> uv_remote.UvSetupPayload:
    spec = make_spec(tmp_path, **overrides)
    return uv_remote.build_setup_payload(
        spec,
        csh_driver_path=str(tmp_path / "remote root" / "launch" / "setup driver.csh"),
        bash_driver_path=str(tmp_path / "remote root" / "launch" / "setup driver.sh"),
    )


def result_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": uv_remote.SCHEMA_VERSION,
        "ok": True,
        "status": "ready",
        "phase": "ready",
        "uv_version": "0.8.14",
        "source_sha256": "a" * 64,
        "lock_sha256": "b" * 64,
        "setup_environment_sha256": "c" * 64,
        "python_request": "3.11.5",
        "python_path": "/tmp/project/env/bin/python",
        "source_dir": "/tmp/project/source",
        "environment_dir": "/tmp/project/env",
        "uv_binary_path": "/tmp/project/tools/uv",
        "ready_state_path": "/tmp/project/state/ready.json",
        "failed_state_path": "/tmp/project/state/failed.json",
        "log_path": "/tmp/project/logs/setup.log",
        "reused_uv": False,
        "reused_environment": False,
        "returncode": 0,
        "error": "",
        "failed_command": "",
        "failed_line": None,
    }
    payload.update(overrides)
    return payload


def sentinel(payload: object) -> str:
    return "\n".join(
        [
            "arbitrary login noise",
            uv_remote.SETUP_SENTINEL_BEGIN,
            json.dumps(payload, separators=(",", ":")),
            uv_remote.SETUP_SENTINEL_END,
            "logout noise",
        ]
    )


def test_build_setup_payload_exposes_structured_files_and_entrypoint(tmp_path: Path) -> None:
    payload = build_payload(tmp_path)

    assert payload.entrypoint == ("csh", "-f", payload.csh_driver_path)
    assert payload.csh_driver_path.endswith("setup driver.csh")
    assert payload.bash_driver_path.endswith("setup driver.sh")
    assert payload.files == {
        payload.csh_driver_path: payload.csh_source,
        payload.bash_driver_path: payload.bash_source,
    }


def test_csh_driver_sources_tsg_then_cuda_restores_gpu_and_execs_clean_bash(tmp_path: Path) -> None:
    payload = build_payload(tmp_path)
    source = payload.csh_source

    python_index = source.index("source /opt/Python/Python-3.11.5_Setup.csh")
    cuda_index = source.index('source "$ucl_cuda_setup"')
    gpu_index = source.index("setenv CUDA_VISIBLE_DEVICES 0")
    exec_index = source.index('exec /bin/bash --noprofile --norc "$ucl_bash_driver"')
    assert python_index < cuda_index < gpu_index < exec_index
    assert "#!/bin/csh -f" in source
    assert "bash -lc" not in source


def test_cpu_csh_driver_does_not_source_cuda_or_set_gpu(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    assert "CUDA_VISIBILITY" not in payload.csh_source
    assert "CUDA_VISIBLE_DEVICES" not in payload.csh_source


def test_setup_uses_exact_versioned_installer_atomic_promotion_and_no_fallbacks(tmp_path: Path) -> None:
    source = build_payload(tmp_path).bash_source

    assert "https://astral.sh/uv/0.8.14/install.sh" in source
    assert "UV_UNMANAGED_INSTALL" in source
    assert 'ucl_uv_version_is_exact "$ucl_uv_candidate"' in source
    assert 'mv -- "$ucl_tool_temp" "$UCL_UV_TOOL_DIR"' in source
    assert "latest" not in source.lower()
    for forbidden in ("pip install", "pip3", "conda", "docker", "UV_NO_MANAGED_PYTHON"):
        assert forbidden not in source


def test_setup_uses_frozen_lock_sync_check_and_managed_python_paths(tmp_path: Path) -> None:
    source = build_payload(tmp_path).bash_source

    assert '"$UCL_UV_BINARY" lock --check' in source
    assert '"$UCL_UV_BINARY" sync --frozen --no-editable' in source
    assert '"$UCL_UV_BINARY" sync --frozen --check' in source
    assert "UV_CACHE_DIR=" in source
    assert "UV_PROJECT_ENVIRONMENT=" in source
    assert "UV_PYTHON_INSTALL_DIR=" in source
    assert "UV_PYTHON_DOWNLOADS" not in source
    assert 'cp -a -- "$UCL_SOURCE_DIR/." "$ucl_build_source/"' in source
    assert 'chmod -R u+w "$ucl_build_source"' in source


def test_setup_has_separate_tool_and_environment_locks_and_atomic_state(tmp_path: Path) -> None:
    source = build_payload(tmp_path).bash_source

    assert "flock -w 1800 9" in source
    assert "flock -w 1800 8" in source
    assert "os.replace(temporary, target)" in source
    assert "UCL_READY_STATE" in source
    assert "UCL_FAILED_STATE" in source
    assert "CURRENT_PHASE=" in source
    assert "trap 'ucl_fail $? $LINENO \"$BASH_COMMAND\"' ERR" in source
    assert "setup cancelled by $signal_name" in source


def test_csh_driver_quotes_paths_without_embedding_them_as_shell_syntax(tmp_path: Path) -> None:
    payload = uv_remote.build_setup_payload(
        make_spec(tmp_path),
        csh_driver_path=str(tmp_path / "remote root" / "launch" / "setup's driver.csh"),
        bash_driver_path=str(tmp_path / "remote root" / "launch" / "setup ! driver.sh"),
    )

    assert "setup ! driver.sh" not in payload.csh_source
    assert "setup's driver" not in payload.csh_source
    assert "base64.b64decode" in payload.csh_source


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_dir", "relative/source"),
        ("source_dir", "/"),
        ("environment_dir", "/tmp/root/../escape"),
        ("uv_cache_dir", "/tmp/cache\nother"),
        ("log_path", "/tmp/log\x00name"),
    ],
)
def test_remote_paths_reject_unsafe_values(tmp_path: Path, field: str, value: str) -> None:
    spec = make_spec(tmp_path)
    paths = spec.paths.__dict__ | {field: value}

    with pytest.raises(ValueError, match=field):
        uv_remote.UvRemotePaths(**paths)


def test_remote_paths_reject_binary_outside_tool_and_destructive_overlap(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    values = spec.paths.__dict__ | {"uv_binary_path": str(tmp_path / "other" / "uv")}
    with pytest.raises(ValueError, match="uv_binary_path.*uv_tool_dir"):
        uv_remote.UvRemotePaths(**values)

    values = spec.paths.__dict__ | {"ready_state_path": str(Path(spec.paths.environment_dir) / "ready.json")}
    with pytest.raises(ValueError, match="ready_state_path.*environment_dir"):
        uv_remote.UvRemotePaths(**values)


@pytest.mark.parametrize("version", ["", "latest", "v0.8.14", "0.8.14; touch /tmp/x"])
def test_setup_spec_rejects_unsafe_or_unpinned_uv_versions(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="uv_version"):
        make_spec(tmp_path, uv_version=version)


def test_setup_spec_requires_numeric_gpu_and_explicit_cuda_script(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gpu_id"):
        make_spec(tmp_path, gpu_id="auto")
    with pytest.raises(ValueError, match="cuda_visibility_script"):
        make_spec(tmp_path, gpu_id="0", cuda_visibility_script=None)
    with pytest.raises(ValueError, match="cuda_visibility_script"):
        make_spec(tmp_path, gpu_id=None, cuda_visibility_script="/tmp/cuda.csh")


def test_driver_paths_must_be_distinct_absolute_safe_paths(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    with pytest.raises(ValueError, match="csh_driver_path"):
        uv_remote.build_setup_payload(spec, csh_driver_path="relative.csh", bash_driver_path="/tmp/setup.sh")
    with pytest.raises(ValueError, match="distinct"):
        uv_remote.build_setup_payload(spec, csh_driver_path="/tmp/setup", bash_driver_path="/tmp/setup")


def test_setup_spec_rejects_identity_and_managed_environment_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_sha256"):
        make_spec(tmp_path, source_sha256="not-a-hash")
    with pytest.raises(ValueError, match="python_request"):
        make_spec(tmp_path, python_request="3.11; rm -rf")
    with pytest.raises(ValueError, match="managed by ucl"):
        make_spec(tmp_path, setup_env=(("UV_CACHE_DIR", "/other"),))


def test_parse_setup_result_ignores_outside_noise_and_preserves_failure_details() -> None:
    payload = result_payload(
        ok=False,
        status="failed",
        phase="sync",
        returncode=17,
        error="uv sync failed exactly: incompatible wheel",
        failed_command="uv sync --frozen --no-editable",
        failed_line=218,
    )

    result = uv_remote.parse_setup_result(sentinel(payload))

    assert result.status == "failed"
    assert result.phase == "sync"
    assert result.returncode == 17
    assert result.error == "uv sync failed exactly: incompatible wheel"
    assert result.failed_command == "uv sync --frozen --no-editable"
    assert result.failed_line == 218


@pytest.mark.parametrize(
    "text,message",
    [
        ("no marker", "not found"),
        (f"{uv_remote.SETUP_SENTINEL_BEGIN}\n{{}}", "end"),
        (
            sentinel(result_payload()) + "\n" + sentinel(result_payload()),
            "exactly once|multiple",
        ),
        (
            f"{uv_remote.SETUP_SENTINEL_BEGIN}\nnot json\n{uv_remote.SETUP_SENTINEL_END}",
            "valid JSON",
        ),
    ],
)
def test_parse_setup_result_rejects_missing_multiple_and_malformed_sentinels(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        uv_remote.parse_setup_result(text)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"schema_version": 2}, "schema"),
        ({"ok": "yes"}, "ok"),
        ({"status": "working"}, "status"),
        ({"phase": 7}, "phase"),
        ({"reused_uv": 1}, "reused_uv"),
        ({"returncode": True}, "returncode"),
        ({"failed_line": "12"}, "failed_line"),
        ({"ok": True, "status": "failed"}, "contradict"),
        ({"ok": False, "status": "ready"}, "contradict"),
        ({"ok": False, "status": "failed", "returncode": 1, "error": ""}, "error"),
    ],
)
def test_state_parser_rejects_bad_schema_types_and_contradictions(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        uv_remote.parse_state_json(json.dumps(result_payload(**overrides)))


def _write_fake_installer(bin_dir: Path) -> None:
    curl = bin_dir / "curl"
    curl.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[-1])
target.write_text('''#!/bin/sh
set -eu
mkdir -p "$UV_UNMANAGED_INSTALL"
cat > "$UV_UNMANAGED_INSTALL/uv" <<'UV'
#!/usr/bin/env python3
import os
import pathlib
import sys
import time

args = sys.argv[1:]
if args == ["--version"]:
    print("uv 0.8.14")
    raise SystemExit(0)
if args[:2] == ["lock", "--check"]:
    raise SystemExit(0)
if args and args[0] == "sync":
    if os.environ.get("FAKE_UV_FAIL") == "1" and "--no-editable" in args:
        print("incompatible wheel", file=sys.stderr)
        raise SystemExit(17)
    if "--no-editable" in args:
        counter = os.environ.get("FAKE_UV_COUNTER")
        if counter:
            with open(counter, "a", encoding="utf-8") as handle:
                handle.write("sync\\\\n")
        time.sleep(float(os.environ.get("FAKE_UV_SLEEP", "0")))
        environment = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"])
        environment.mkdir(parents=True, exist_ok=True)
        (environment / "bin").mkdir(exist_ok=True)
        interpreter = environment / "bin" / "python"
        if not interpreter.exists():
            interpreter.symlink_to(sys.executable)
    if "--check" in args and not pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"]).is_dir():
        raise SystemExit(9)
    raise SystemExit(0)
print("unexpected uv argv: " + repr(args), file=sys.stderr)
raise SystemExit(99)
UV
chmod +x "$UV_UNMANAGED_INSTALL/uv"
''', encoding='utf-8')
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    flock = bin_dir / "flock"
    flock.write_text(
        """#!/usr/bin/env python3
import fcntl
import sys

fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX)
""",
        encoding="utf-8",
    )
    flock.chmod(0o755)


def _write_project_contract(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "pyproject.toml").write_text("[project]\nname='smoke'\nversion='0.0.0'\n", encoding="utf-8")
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / ".python-version").write_text("3.11.5\n", encoding="utf-8")


def _run_bash_payload(
    payload: uv_remote.UvSetupPayload,
    tmp_path: Path,
    *,
    fail: bool = False,
    counter: Path | None = None,
    sleep: float = 0,
    readonly_source: bool = False,
    tamper_lock: bool = False,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    _write_fake_installer(bin_dir)
    source_dir = Path(payload.spec.paths.source_dir)
    _write_project_contract(source_dir)
    if tamper_lock:
        (source_dir / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    if readonly_source:
        for path in source_dir.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        source_dir.chmod(0o555)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    if fail:
        env["FAKE_UV_FAIL"] = "1"
    if counter is not None:
        env["FAKE_UV_COUNTER"] = str(counter)
    env["FAKE_UV_SLEEP"] = str(sleep)
    try:
        return subprocess.run(
            ["/bin/bash", "--noprofile", "--norc"],
            input=payload.bash_source,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
            check=False,
        )
    finally:
        if readonly_source:
            source_dir.chmod(0o755)
            for path in source_dir.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif not path.is_symlink():
                    path.chmod(0o644)


def test_generated_setup_driver_succeeds_then_reuses_exact_tool_and_environment(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    first = _run_bash_payload(payload, tmp_path)
    second = _run_bash_payload(payload, tmp_path)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_result = uv_remote.parse_setup_result(first.stdout)
    second_result = uv_remote.parse_setup_result(second.stdout)
    assert not first_result.reused_uv
    assert not first_result.reused_environment
    assert second_result.reused_uv
    assert second_result.reused_environment
    ready = uv_remote.parse_state_json(Path(payload.spec.paths.ready_state_path).read_text(encoding="utf-8"))
    assert ready.ok
    assert not Path(payload.spec.paths.failed_state_path).exists()


def test_generated_setup_driver_builds_from_readonly_immutable_source(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    process = _run_bash_payload(payload, tmp_path, readonly_source=True)

    assert process.returncode == 0, process.stdout
    assert uv_remote.parse_setup_result(process.stdout).ok


def test_generated_setup_driver_writes_exact_structured_failure_and_no_ready_marker(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    proc = _run_bash_payload(payload, tmp_path, fail=True)

    assert proc.returncode == 17
    result = uv_remote.parse_setup_result(proc.stdout)
    assert not result.ok
    assert result.phase == "sync"
    assert result.returncode == 17
    assert "sync" in result.failed_command
    assert "exit 17" in result.error
    failed = uv_remote.parse_state_json(Path(payload.spec.paths.failed_state_path).read_text(encoding="utf-8"))
    assert failed.error == result.error
    assert not Path(payload.spec.paths.ready_state_path).exists()


def test_generated_setup_driver_rejects_a_changed_lock_before_bootstrap(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    process = _run_bash_payload(payload, tmp_path, tamper_lock=True)

    assert process.returncode != 0
    result = uv_remote.parse_setup_result(process.stdout)
    assert result.phase == "preflight"
    assert "uv.lock digest mismatch" in result.error
    assert not Path(payload.spec.paths.uv_binary_path).exists()


def test_setup_failure_handlers_do_not_remove_another_jobs_ready_state(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)

    assert 'rm -f -- "$UCL_READY_STATE"' not in payload.bash_source
    assert "trap 'ucl_cancel 129 HUP' HUP" in payload.bash_source
    assert not Path(payload.spec.paths.environment_dir).exists()


def test_concurrent_setup_is_serialized_and_only_syncs_environment_once(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, gpu_id=None, cuda_visibility_script=None)
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    _write_fake_installer(bin_dir)
    _write_project_contract(Path(payload.spec.paths.source_dir))
    counter = tmp_path / "sync-counter"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_UV_COUNTER": str(counter),
            "FAKE_UV_SLEEP": "0.2",
        }
    )

    processes = [
        subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(2)
    ]
    outputs = [process.communicate(payload.bash_source, timeout=15) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], outputs
    results = [uv_remote.parse_setup_result(stdout) for stdout, _ in outputs]
    assert sorted(result.reused_environment for result in results) == [False, True]
    assert counter.read_text(encoding="utf-8").splitlines() == ["sync"]
