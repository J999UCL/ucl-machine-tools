from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def import_toolkit() -> tuple[Any, Any, Any]:
    from ucl_machine_tools import hosts, inventory
    from ucl_machine_tools import main_cli

    return hosts, inventory, main_cli


def ok(stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def fail(stdout: str = "", stderr: str = "failed") -> SimpleNamespace:
    return SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr)


def make_catalog(remote_hosts: Any) -> dict[str, Any]:
    return remote_hosts.validate_catalog(
        [
            remote_hosts.HostSpec(
                name="barbury-l",
                ssh_host="barbury-l",
                labels=("ucl", "ucl-gpu", "cuda"),
                scratch_root="/tmp/ucl-machine-tools",
                restart_policy="lab_pc",
            ),
            remote_hosts.HostSpec(
                name="barbury-m",
                ssh_host="barbury-m",
                labels=("ucl", "ucl-gpu", "cuda"),
                scratch_root="/tmp/ucl-machine-tools",
                restart_policy="lab_pc",
            ),
            remote_hosts.HostSpec(
                name="login-cpu",
                ssh_host="knuckles",
                labels=("ucl", "login"),
                scratch_root="/tmp/ucl-machine-tools",
                restart_policy="unknown",
            ),
        ]
    )


def inventory_payload(
    *,
    host: str = "barbury-l",
    ok: bool = True,
    gpus: list[dict[str, Any]] | None = None,
    filesystems: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "host": host,
        "hostname": f"{host}.ucl.example",
        "ok": ok,
        "gpus": gpus if gpus is not None else [],
        "filesystems": filesystems if filesystems is not None else [],
        "scratch": {"root": "/tmp/ucl-machine-tools", "exists": True},
        "restart": {
            "policy": "lab_pc",
            "text": "Mon/Thu 19:30-midnight; may reboot anytime",
        },
        "errors": errors if errors is not None else [],
    }


def gpu(index: int = 0, *, processes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "index": index,
        "name": "NVIDIA GeForce RTX 4060 Ti",
        "memory_total_mb": 8192,
        "memory_used_mb": 1024,
        "utilization_gpu_percent": 7,
        "processes": processes if processes is not None else [],
    }


def tmp_fs(available_gb: float = 512.0) -> dict[str, Any]:
    return {"path": "/tmp", "available_gb": available_gb, "used_percent": 31.5}


def sentinel_stdout(remote_inventory: Any, payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Last login: Tue Jul 7 10:12:00 2026 from 10.0.0.7",
            "Module warning: CUDA visibility script already sourced",
            remote_inventory.INVENTORY_SENTINEL_BEGIN,
            json.dumps(payload, sort_keys=True),
            remote_inventory.INVENTORY_SENTINEL_END,
            "Connection to barbury-l closed.",
        ]
    )


def test_default_catalog_validates_known_ucl_gpu_host() -> None:
    remote_hosts, _, _ = import_toolkit()

    catalog = remote_hosts.validate_catalog(remote_hosts.DEFAULT_CATALOG)

    assert "barbury-l" in catalog
    assert catalog["barbury-l"].ssh_host == "barbury-l"
    assert "ucl-gpu" in catalog["barbury-l"].labels
    assert catalog["barbury-l"].scratch_root.startswith("/tmp/")


def test_catalog_validation_rejects_duplicate_names_and_unsafe_ssh_hosts() -> None:
    remote_hosts, _, _ = import_toolkit()

    with pytest.raises(ValueError, match="duplicate"):
        remote_hosts.validate_catalog(
            [
                remote_hosts.HostSpec(name="barbury-l", ssh_host="barbury-l", labels=("ucl-gpu",)),
                remote_hosts.HostSpec(name="barbury-l", ssh_host="barbury-l-alt", labels=("ucl-gpu",)),
            ]
        )

    with pytest.raises(ValueError, match="ssh_host"):
        remote_hosts.validate_catalog(
            [remote_hosts.HostSpec(name="barbury-l", ssh_host="barbury-l; rm -rf /", labels=("ucl-gpu",))]
        )

    with pytest.raises(ValueError, match="scratch_root"):
        remote_hosts.validate_catalog(
            [remote_hosts.HostSpec(name="barbury-l", ssh_host="barbury-l", labels=("ucl-gpu",), scratch_root="tmp")]
        )


def test_selector_parser_expands_labels_names_and_exclusions_in_catalog_order() -> None:
    remote_hosts, _, _ = import_toolkit()
    catalog = make_catalog(remote_hosts)

    selected = remote_hosts.parse_selector("label:ucl-gpu,!barbury-m", catalog=catalog)

    assert [host.name for host in selected] == ["barbury-l"]
    assert [host.name for host in remote_hosts.parse_selector("barbury-m,barbury-l", catalog=catalog)] == [
        "barbury-l",
        "barbury-m",
    ]
    assert [host.name for host in remote_hosts.parse_selector("all,!label:login", catalog=catalog)] == [
        "barbury-l",
        "barbury-m",
    ]


def test_default_catalog_selector_supports_gpu_classes_groups_and_aliases() -> None:
    remote_hosts, _, _ = import_toolkit()
    catalog = remote_hosts.load_catalog()

    assert remote_hosts.parse_selector("timeshare", catalog=catalog)[0].name == "cream"
    assert "barbury-l" in [host.name for host in remote_hosts.parse_selector("3090ti", catalog=catalog)]
    assert remote_hosts.parse_selector("pocher-l", catalog=catalog)[0].name == "pochard-l"


@pytest.mark.parametrize("selector", ["", "unknown-host", "label:missing", "all,!all"])
def test_selector_parser_rejects_empty_unknown_and_empty_results(selector: str) -> None:
    remote_hosts, _, _ = import_toolkit()
    catalog = make_catalog(remote_hosts)

    with pytest.raises(ValueError):
        remote_hosts.parse_selector(selector, catalog=catalog)


def test_ssh_argv_construction_is_list_argv_with_no_local_shell_tokens() -> None:
    remote_hosts, remote_inventory, _ = import_toolkit()
    host = make_catalog(remote_hosts)["barbury-l"]

    argv = remote_inventory.build_ssh_argv(host)

    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    assert argv[0] == "ssh"
    assert "barbury-l" in argv
    assert argv[-2:] == ["python3", "-"]
    assert "shell=True" not in argv
    assert not any(part in {";", "|", "&&"} for part in argv)

    source = remote_inventory.build_remote_probe_source(host=host, root="/tmp/ucl-machine-tools", sizes=False)
    compile(source, "<remote_inventory_probe>", "exec")
    assert source.split("PARAMS =", 1)[1].splitlines()[0].lstrip().startswith("json.loads(")


def test_collect_uses_fake_runner_with_list_argv_and_no_shell() -> None:
    remote_hosts, remote_inventory, _ = import_toolkit()
    host = make_catalog(remote_hosts)["barbury-l"]
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        assert isinstance(argv, list)
        assert kwargs.get("shell", False) is False
        return SimpleNamespace(
            returncode=0,
            stdout=sentinel_stdout(
                remote_inventory,
                inventory_payload(host="barbury-l", gpus=[gpu()], filesystems=[tmp_fs()]),
            ),
            stderr="",
        )

    rows = remote_inventory.collect([host], runner=fake_runner)

    assert calls
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert rows[0]["host"] == "barbury-l"
    assert rows[0]["status"] == "ready"


def test_sentinel_parser_ignores_login_noise_and_rejects_missing_sentinel() -> None:
    _, remote_inventory, _ = import_toolkit()
    payload = inventory_payload(host="barbury-l", gpus=[gpu()], filesystems=[tmp_fs()])

    parsed = remote_inventory.parse_sentinel_stdout(sentinel_stdout(remote_inventory, payload))

    assert parsed["host"] == "barbury-l"
    assert parsed["gpus"][0]["name"] == "NVIDIA GeForce RTX 4060 Ti"
    assert parsed["filesystems"][0]["path"] == "/tmp"
    with pytest.raises(ValueError, match="sentinel"):
        remote_inventory.parse_sentinel_stdout(json.dumps(payload))


def test_missing_sentinel_does_not_expose_login_noise_unless_debug() -> None:
    remote_hosts, remote_inventory, _ = import_toolkit()
    host = make_catalog(remote_hosts)["barbury-l"]

    def fake_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=255, stdout="Last login: noisy", stderr="VBoxManage noisy stderr")

    clean = remote_inventory.collect([host], runner=fake_runner, debug=False)[0]
    noisy = remote_inventory.collect([host], runner=fake_runner, debug=True)[0]

    assert "stderr_tail" not in clean
    assert "stdout_tail" not in clean
    assert noisy["stderr_tail"] == "VBoxManage noisy stderr"


def test_collect_one_distinguishes_unreachable_ssh_from_missing_sentinel() -> None:
    remote_hosts, remote_inventory, _ = import_toolkit()
    host = make_catalog(remote_hosts)["barbury-l"]

    unreachable = remote_inventory.collect_one(
        host,
        runner=lambda argv, **kwargs: SimpleNamespace(
            returncode=255,
            stdout="",
            stderr="ssh: connect to host barbury-l: No route to host",
        ),
    )
    missing_sentinel = remote_inventory.collect_one(
        host,
        runner=lambda argv, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Last login: noisy",
            stderr="",
        ),
    )

    assert unreachable["status"] == "unreachable"
    assert unreachable["ssh_returncode"] == 255
    assert unreachable["errors"] == ["target host is unreachable from the jump host"]
    assert "sentinel" not in unreachable["errors"][0]
    assert missing_sentinel["status"] == "no-sentinel"
    assert missing_sentinel["ssh_returncode"] == 0
    assert missing_sentinel["errors"] == ["inventory sentinel not found"]


def test_classification_distinguishes_ready_busy_no_gpu_and_unreachable() -> None:
    _, remote_inventory, _ = import_toolkit()
    busy_process = {"pid": 4242, "user": "other", "used_memory_mb": 7000, "command": "python train.py"}

    assert remote_inventory.classify(inventory_payload(gpus=[gpu()], filesystems=[tmp_fs(600.0)])) == "ready"
    assert (
        remote_inventory.classify(inventory_payload(gpus=[gpu(processes=[busy_process])], filesystems=[tmp_fs(600.0)]))
        == "busy"
    )
    assert remote_inventory.classify(inventory_payload(gpus=[], filesystems=[tmp_fs(600.0)])) == "no-gpu"
    assert (
        remote_inventory.classify(inventory_payload(ok=False, errors=["ssh exited 255"], filesystems=[]))
        == "unreachable"
    )
    assert remote_inventory.classify(inventory_payload(gpus=[gpu()], filesystems=[tmp_fs(2.0)])) == "storage-low"


def test_human_table_formatting_is_plain_text_and_scan_friendly() -> None:
    _, remote_inventory, _ = import_toolkit()
    rows = [
        {**inventory_payload(host="barbury-l", gpus=[gpu()], filesystems=[tmp_fs(512.0)]), "status": "ready"},
        {
            **inventory_payload(
                host="barbury-m",
                gpus=[gpu(processes=[{"pid": 111, "user": "example-user", "used_memory_mb": 4096}])],
                filesystems=[tmp_fs(88.5)],
            ),
            "status": "busy",
        },
    ]

    table = remote_inventory.format_table(rows)
    lines = table.splitlines()

    assert {"host", "status", "gpu", "tmp_free", "tmp_scratch", "restart", "ssh"}.issubset(
        set(lines[0].lower().split())
    )
    assert "barbury-l" in table
    assert "ready" in table
    assert "barbury-m" in table
    assert "busy" in table
    assert "RTX 4060 Ti" in table
    assert "Mon/Thu 19:30-midnight" in table
    assert "syn4d" not in table.lower()
    assert "mvtracker" not in table.lower()
    assert "omega" not in table.lower()
    assert "fpt" not in table.lower()
    assert "{" not in table
    assert "\x1b" not in table


def test_json_shape_is_stable_and_counts_statuses() -> None:
    _, remote_inventory, _ = import_toolkit()
    rows = [
        {**inventory_payload(host="barbury-l", gpus=[gpu()], filesystems=[tmp_fs(512.0)]), "status": "ready"},
        {**inventory_payload(host="barbury-m", gpus=[gpu()], filesystems=[tmp_fs(2.0)]), "status": "storage-low"},
        {**inventory_payload(host="login-cpu", ok=False, errors=["ssh exited 255"]), "status": "unreachable"},
    ]

    payload = remote_inventory.to_jsonable(rows)

    json.dumps(payload, sort_keys=True)
    assert payload["schema_version"] == 1
    assert [host["host"] for host in payload["hosts"]] == ["barbury-l", "barbury-m", "login-cpu"]
    assert payload["summary"] == {"ready": 1, "storage-low": 1, "total": 3, "unreachable": 1}
    assert set(payload["hosts"][0]) >= {"host", "status", "gpus", "filesystems", "errors"}


def test_cli_json_uses_fake_runner_only_and_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    _, remote_inventory, ucl_inventory = import_toolkit()
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((argv, kwargs))
        assert isinstance(argv, list)
        assert kwargs.get("shell", False) is False
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        assert "barbury-l" in argv
        return ok(
            stdout=sentinel_stdout(
                remote_inventory,
                inventory_payload(host="barbury-l", gpus=[gpu()], filesystems=[tmp_fs(700.0)]),
            ),
        )

    assert ucl_inventory.main(["status", "--selector", "barbury-l", "--json"], runner=fake_runner) == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert calls
    assert calls[0][0] == ["ssh", "-O", "check", "knuckles"]
    assert payload["schema_version"] == 1
    assert payload["hosts"][0]["host"] == "barbury-l"
    assert payload["hosts"][0]["status"] == "ready"


def test_cli_subcommands_filter_and_recommend_without_remote_noise(capsys: pytest.CaptureFixture[str]) -> None:
    _, remote_inventory, ucl_inventory = import_toolkit()

    def fake_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return ok()
        host = "barbury-l" if "barbury-l" in argv else "canada-l"
        row_gpu = gpu()
        if host == "canada-l":
            row_gpu = gpu(processes=[{"pid": 111, "user": "busy", "used_memory_mb": 6000}])
        return ok(
            stdout=sentinel_stdout(
                remote_inventory,
                inventory_payload(host=host, gpus=[row_gpu], filesystems=[tmp_fs(700.0)]),
            ),
            stderr="loud login noise that must not be printed",
        )

    assert ucl_inventory.main(["status", "recommend", "barbury-l,canada-l", "--json"], runner=fake_runner) == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert [row["host"] for row in payload["hosts"]] == ["barbury-l"]
    assert "loud login noise" not in out


def test_cli_help_documents_selectors_and_output_modes(capsys: pytest.CaptureFixture[str]) -> None:
    _, _, ucl_inventory = import_toolkit()
    parser = ucl_inventory.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "status" in help_text
    assert "exec" in help_text
    assert "run" in help_text
    assert "--use-master" not in help_text
    assert "ucl" in help_text.lower()

    status_parser = ucl_inventory.build_parser()
    with pytest.raises(SystemExit):
        status_parser.parse_args(["status", "--help"])
    status_help = capsys.readouterr().out
    assert "--selector" in status_help
    assert "--json" in status_help
    assert "--table" in status_help


def test_ssh_master_helper_checks_and_starts_when_needed() -> None:
    from ucl_machine_tools import ssh

    calls: list[list[str]] = []

    def existing_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        return ok()

    assert ssh.ensure_knuckles_master(runner=existing_runner) == "existing"
    assert calls == [["ssh", "-O", "check", "knuckles"]]

    calls.clear()

    def start_runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        if argv[:3] == ["ssh", "-O", "check"]:
            return fail(stderr="No such control socket")
        return ok()

    assert ssh.ensure_knuckles_master(runner=start_runner) == "started"
    assert calls == [["ssh", "-O", "check", "knuckles"], ["ssh", "-MNf", "knuckles"]]


def test_ssh_master_start_failure_is_clear() -> None:
    from ucl_machine_tools import ssh

    def runner(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        if argv[:3] == ["ssh", "-O", "check"]:
            return fail(stderr="missing")
        return fail(stderr="permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        ssh.ensure_knuckles_master(runner=runner)
