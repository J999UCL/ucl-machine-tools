from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ucl_machine_tools.uv_project import (
    UvProjectError,
    build_source_manifest,
    check_uv_lock,
    derive_remote_layout,
    discover_local_uv,
    hash_setup_environment,
    load_uv_project,
    materialize_source_snapshot,
    validate_uv_project,
)


def make_project(root: Path, *, python_request: str = "3.11.5") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")
    (root / ".python-version").write_text(f"{python_request}\n", encoding="utf-8")
    return root


def make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class RecordingRunner:
    def __init__(self, results: list[subprocess.CompletedProcess[str]]) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        return self.results.pop(0)


@pytest.mark.parametrize("missing", ["pyproject.toml", "uv.lock", ".python-version"])
def test_validate_uv_project_requires_complete_contract(tmp_path: Path, missing: str) -> None:
    project = make_project(tmp_path / "project")
    (project / missing).unlink()

    with pytest.raises(UvProjectError, match=missing.replace(".", r"\.")):
        validate_uv_project(project)


def test_validate_uv_project_parses_contract_and_hashes_lock(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project", python_request="cpython-3.11.5")

    contract = validate_uv_project(project)

    assert contract.root == project.resolve()
    assert contract.pyproject_path == project.resolve() / "pyproject.toml"
    assert contract.lock_path == project.resolve() / "uv.lock"
    assert contract.python_version_path == project.resolve() / ".python-version"
    assert contract.python_request == "cpython-3.11.5"
    assert len(contract.lock_sha256) == 64


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("pyproject.toml", "not = [toml", "invalid pyproject.toml"),
        ("uv.lock", "version = [", "invalid uv.lock"),
        (".python-version", "", "one Python request"),
        (".python-version", "3.11\n3.12\n", "one Python request"),
        (".python-version", "3.11; touch nope\n", "unsafe Python request"),
    ],
)
def test_validate_uv_project_rejects_malformed_contract(
    tmp_path: Path,
    filename: str,
    contents: str,
    message: str,
) -> None:
    project = make_project(tmp_path / "project")
    (project / filename).write_text(contents, encoding="utf-8")

    with pytest.raises(UvProjectError, match=message):
        validate_uv_project(project)


def test_discover_local_uv_returns_exact_executable_and_version(tmp_path: Path) -> None:
    executable = make_executable(tmp_path / "bin" / "uv")
    runner = RecordingRunner(
        [subprocess.CompletedProcess([str(executable), "--version"], 0, "uv 0.9.27 (Homebrew 2026-01-26)\n", "")]
    )

    tool = discover_local_uv(runner=runner, which=lambda name: str(executable))

    assert tool.executable == executable.resolve()
    assert tool.version == "0.9.27"
    assert runner.calls == [
        (
            [str(executable.resolve()), "--version"],
            {"capture_output": True, "text": True, "check": False},
        )
    ]


def test_discover_local_uv_requires_an_executable_and_valid_version(tmp_path: Path) -> None:
    with pytest.raises(UvProjectError, match="not found"):
        discover_local_uv(which=lambda name: None)

    executable = make_executable(tmp_path / "uv")
    runner = RecordingRunner([subprocess.CompletedProcess([], 0, "surprising output\n", "")])
    with pytest.raises(UvProjectError, match="could not parse"):
        discover_local_uv(runner=runner, which=lambda name: str(executable))


def test_check_uv_lock_uses_exact_binary_and_project_argv(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    contract = validate_uv_project(project)
    executable = make_executable(tmp_path / "uv")
    tool_runner = RecordingRunner([subprocess.CompletedProcess([], 0, "uv 0.9.27\n", "")])
    tool = discover_local_uv(runner=tool_runner, which=lambda name: str(executable))
    runner = RecordingRunner([subprocess.CompletedProcess([], 0, "", "")])

    check_uv_lock(contract, tool, runner=runner)

    assert runner.calls == [
        (
            [str(executable.resolve()), "lock", "--check", "--project", str(project.resolve())],
            {"capture_output": True, "text": True, "check": False},
        )
    ]


def test_check_uv_lock_reports_stale_lock_without_hiding_uv_error(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    contract = validate_uv_project(project)
    executable = make_executable(tmp_path / "uv")
    tool = discover_local_uv(
        runner=RecordingRunner([subprocess.CompletedProcess([], 0, "uv 0.9.27\n", "")]),
        which=lambda name: str(executable),
    )
    runner = RecordingRunner([subprocess.CompletedProcess([], 2, "", "The lockfile needs to be updated\n")])

    with pytest.raises(UvProjectError, match="The lockfile needs to be updated"):
        check_uv_lock(contract, tool, runner=runner)


def test_load_uv_project_checks_lock_before_building_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = make_project(tmp_path / "project")
    executable = make_executable(tmp_path / "uv")
    runner = RecordingRunner(
        [
            subprocess.CompletedProcess([], 0, "uv 0.9.27\n", ""),
            subprocess.CompletedProcess([], 2, "", "stale lock"),
        ]
    )
    called = False

    def forbidden_manifest(root: Path):
        nonlocal called
        called = True
        raise AssertionError("manifest should not run after a failed lock check")

    monkeypatch.setattr("ucl_machine_tools.uv_project.build_source_manifest", forbidden_manifest)

    with pytest.raises(UvProjectError, match="stale lock"):
        load_uv_project(project, runner=runner, which=lambda name: str(executable))
    assert not called


def test_manifest_is_deterministic_and_tracks_content_and_executable_mode(tmp_path: Path) -> None:
    first = make_project(tmp_path / "first")
    second = make_project(tmp_path / "second")
    for root in (first, second):
        (root / "pkg").mkdir()
        (root / "pkg" / "b.py").write_text("print('b')\n", encoding="utf-8")
        (root / "a.py").write_text("print('a')\n", encoding="utf-8")
    os.utime(second / "a.py", ns=(1_000_000_000, 1_000_000_000))

    first_manifest = build_source_manifest(first)
    second_manifest = build_source_manifest(second)

    assert first_manifest.source_sha256 == second_manifest.source_sha256
    assert [entry.path for entry in first_manifest.entries] == sorted(entry.path for entry in first_manifest.entries)
    assert {entry.path for entry in first_manifest.entries} >= {
        ".python-version",
        "a.py",
        "pkg/b.py",
        "pyproject.toml",
        "uv.lock",
    }

    (second / "a.py").chmod((second / "a.py").stat().st_mode | stat.S_IXUSR)
    executable_manifest = build_source_manifest(second)
    assert executable_manifest.source_sha256 != first_manifest.source_sha256
    assert next(entry for entry in executable_manifest.entries if entry.path == "a.py").executable

    (second / "a.py").write_text("print('changed')\n", encoding="utf-8")
    assert build_source_manifest(second).source_sha256 != executable_manifest.source_sha256


def test_manifest_honors_nested_gitignore_and_uclignore_rules(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    (project / ".gitignore").write_text("*.tmp\n!keep.tmp\nignored/\n", encoding="utf-8")
    (project / ".uclignore").write_text("private/*.txt\n!private/keep.txt\n", encoding="utf-8")
    (project / "drop.tmp").write_text("drop", encoding="utf-8")
    (project / "keep.tmp").write_text("keep", encoding="utf-8")
    (project / "ignored").mkdir()
    (project / "ignored" / ".gitignore").write_text("!hidden.txt\n", encoding="utf-8")
    (project / "ignored" / "hidden.txt").write_text("hidden", encoding="utf-8")
    (project / "nested").mkdir()
    (project / "nested" / ".gitignore").write_text("*.bin\n!keep.bin\n", encoding="utf-8")
    (project / "nested" / "drop.bin").write_text("drop", encoding="utf-8")
    (project / "nested" / "keep.bin").write_text("keep", encoding="utf-8")
    (project / "private").mkdir()
    (project / "private" / "drop.txt").write_text("drop", encoding="utf-8")
    (project / "private" / "keep.txt").write_text("keep", encoding="utf-8")

    paths = {entry.path for entry in build_source_manifest(project).entries}

    assert {".gitignore", ".uclignore", "keep.tmp", "nested/.gitignore", "nested/keep.bin", "private/keep.txt"} <= paths
    assert {"drop.tmp", "ignored/.gitignore", "ignored/hidden.txt", "nested/drop.bin", "private/drop.txt"}.isdisjoint(paths)


def test_mandatory_exclusions_cannot_be_negated_and_env_example_is_allowed(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    (project / ".uclignore").write_text(
        "!.env\n!.env.secret\n!data/keep.txt\n!nested/__pycache__/keep.pyc\n",
        encoding="utf-8",
    )
    (project / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (project / ".env.secret").write_text("TOKEN=secret", encoding="utf-8")
    (project / ".env.example").write_text("TOKEN=", encoding="utf-8")
    (project / "data").mkdir()
    (project / "data" / "keep.txt").write_text("data", encoding="utf-8")
    (project / "nested" / "__pycache__").mkdir(parents=True)
    (project / "nested" / "__pycache__" / "keep.pyc").write_bytes(b"cache")
    (project / ".venv").mkdir()
    (project / ".venv" / "python").write_text("venv", encoding="utf-8")
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("git", encoding="utf-8")

    paths = {entry.path for entry in build_source_manifest(project).entries}

    assert ".env.example" in paths
    assert {".env", ".env.secret", "data/keep.txt", "nested/__pycache__/keep.pyc", ".venv/python", ".git/config"}.isdisjoint(paths)


def test_manifest_rejects_ignored_contract_files_and_preserves_empty_directories(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    (project / "empty").mkdir()
    manifest = build_source_manifest(project)
    assert next(entry for entry in manifest.entries if entry.path == "empty").kind == "directory"

    snapshot = materialize_source_snapshot(manifest)
    try:
        assert (snapshot.temporary_root / "empty").is_dir()
    finally:
        snapshot.cleanup()

    (project / ".gitignore").write_text("uv.lock\n", encoding="utf-8")
    with pytest.raises(UvProjectError, match="required UV contract file is excluded.*uv.lock"):
        build_source_manifest(project)


def test_materialized_snapshot_rejects_a_tree_changed_after_manifest(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    source = project / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = build_source_manifest(project)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(UvProjectError, match="source changed while staging: module.py"):
        materialize_source_snapshot(manifest)


def test_manifest_walks_checked_out_submodule_contents(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    submodule = project / "vendor" / "library"
    submodule.mkdir(parents=True)
    (submodule / ".git").write_text("gitdir: ../../../.git/modules/library\n", encoding="utf-8")
    (submodule / ".gitignore").write_text("generated.txt\n", encoding="utf-8")
    (submodule / "library.py").write_text("VALUE = 1\n", encoding="utf-8")
    (submodule / "generated.txt").write_text("generated", encoding="utf-8")

    paths = {entry.path for entry in build_source_manifest(project).entries}

    assert "vendor/library/library.py" in paths
    assert "vendor/library/.gitignore" in paths
    assert "vendor/library/.git" not in paths
    assert "vendor/library/generated.txt" not in paths


def test_manifest_hashes_safe_symlink_target_and_rejects_escape(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    (project / "targets").mkdir()
    (project / "targets" / "one.txt").write_text("one", encoding="utf-8")
    (project / "targets" / "two.txt").write_text("two", encoding="utf-8")
    link = project / "current.txt"
    link.symlink_to("targets/one.txt")

    first = build_source_manifest(project)
    link_entry = next(entry for entry in first.entries if entry.path == "current.txt")
    assert link_entry.kind == "symlink"
    assert link_entry.symlink_target == "targets/one.txt"

    link.unlink()
    link.symlink_to("targets/two.txt")
    assert build_source_manifest(project).source_sha256 != first.source_sha256

    link.unlink()
    link.symlink_to("../../outside")
    with pytest.raises(UvProjectError, match="escapes project root"):
        build_source_manifest(project)


def test_manifest_rejects_special_files(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    fifo = project / "events.fifo"
    os.mkfifo(fifo)

    with pytest.raises(UvProjectError, match=r"unsupported special file.*events\.fifo"):
        build_source_manifest(project)


def test_load_uv_project_returns_validated_tool_contract_and_manifest(tmp_path: Path) -> None:
    project = make_project(tmp_path / "project")
    (project / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    executable = make_executable(tmp_path / "uv")
    runner = RecordingRunner(
        [
            subprocess.CompletedProcess([], 0, "uv 0.9.27\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )

    spec = load_uv_project(project, runner=runner, which=lambda name: str(executable))

    assert spec.contract.root == project.resolve()
    assert spec.uv.version == "0.9.27"
    assert "demo.py" in {entry.path for entry in spec.manifest.entries}
    assert spec.source_sha256 == spec.manifest.source_sha256
    assert spec.lock_sha256 == spec.contract.lock_sha256


def test_remote_layout_is_safe_content_addressed_and_deterministic() -> None:
    lock_hash = "a" * 64
    source_hash = "b" * 64

    layout = derive_remote_layout(
        remote_root="/tmp/thakwani/fpt",
        stage_name="fpt",
        host="barbury-l",
        uv_version="0.9.27",
        lock_sha256=lock_hash,
        source_sha256=source_hash,
        setup_environment_sha256="c" * 64,
    )
    repeated = derive_remote_layout(
        remote_root="/tmp/thakwani/fpt",
        stage_name="fpt",
        host="barbury-l",
        uv_version="0.9.27",
        lock_sha256=lock_hash,
        source_sha256=source_hash,
        setup_environment_sha256="c" * 64,
    )

    assert layout == repeated
    assert layout.remote_root == PurePosixPath("/tmp/thakwani/fpt")
    assert layout.uv_binary == PurePosixPath("/tmp/thakwani/fpt/tools/uv/0.9.27/uv")
    assert layout.python_install_dir == PurePosixPath("/tmp/thakwani/fpt/tools/python")
    assert layout.uv_cache_dir == PurePosixPath("/tmp/thakwani/fpt/cache/uv")
    assert layout.source_dir == PurePosixPath(f"/tmp/thakwani/fpt/stages/fpt/sources/{source_hash}")
    assert layout.environment_dir == PurePosixPath(f"/tmp/thakwani/fpt/stages/fpt/envs/{layout.environment_id}")
    assert layout.state_file == PurePosixPath(f"/tmp/thakwani/fpt/stages/fpt/state/{layout.stage_id}.json")
    assert layout.launchers_dir == PurePosixPath("/tmp/thakwani/fpt/launchers")
    assert layout.stage_id.startswith("fpt-barbury-l-")
    assert len(layout.environment_id) == 64


def test_remote_identity_changes_for_uv_lock_or_source() -> None:
    base = dict(
        remote_root="/tmp/project",
        stage_name="demo",
        host="canada-l",
        uv_version="0.9.27",
        lock_sha256="a" * 64,
        source_sha256="b" * 64,
        setup_environment_sha256="e" * 64,
    )
    identities = {
        derive_remote_layout(**base).environment_id,
        derive_remote_layout(**{**base, "uv_version": "0.9.28"}).environment_id,
        derive_remote_layout(**{**base, "lock_sha256": "c" * 64}).environment_id,
        derive_remote_layout(**{**base, "source_sha256": "d" * 64}).environment_id,
    }
    assert len(identities) == 4


def test_remote_identity_includes_setup_environment_and_remote_root() -> None:
    base = dict(
        remote_root="/tmp/project-a",
        stage_name="demo",
        host="canada-l",
        uv_version="0.9.27",
        lock_sha256="a" * 64,
        source_sha256="b" * 64,
        setup_environment_sha256=hash_setup_environment((("BUILD_FLAG", "one"),)),
    )
    first = derive_remote_layout(**base)
    different_env = derive_remote_layout(
        **{**base, "setup_environment_sha256": hash_setup_environment((("BUILD_FLAG", "two"),))}
    )
    different_root = derive_remote_layout(**{**base, "remote_root": "/tmp/project-b"})

    assert different_env.environment_id != first.environment_id
    assert different_env.stage_id != first.stage_id
    assert different_root.environment_id == first.environment_id
    assert different_root.stage_id != first.stage_id


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"remote_root": "relative/path"}, "remote_root must be absolute"),
        ({"remote_root": "/tmp/../etc"}, "remote_root must not contain"),
        ({"remote_root": "/"}, "remote_root must not be filesystem root"),
        ({"stage_name": "bad/name"}, "unsafe stage name"),
        ({"host": "host;rm"}, "unsafe host"),
        ({"uv_version": "0.9/27"}, "unsafe uv version"),
        ({"lock_sha256": "short"}, "lock_sha256"),
        ({"source_sha256": "G" * 64}, "source_sha256"),
    ],
)
def test_remote_layout_rejects_unsafe_inputs(changes: dict[str, str], message: str) -> None:
    args = {
        "remote_root": "/tmp/project",
        "stage_name": "demo",
        "host": "barbury-l",
        "uv_version": "0.9.27",
        "lock_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "setup_environment_sha256": "c" * 64,
    }
    args.update(changes)

    with pytest.raises(UvProjectError, match=message):
        derive_remote_layout(**args)
