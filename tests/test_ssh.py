from __future__ import annotations

from ucl_machine_tools import launch
from ucl_machine_tools import rsync_transport
from ucl_machine_tools import ssh as ssh_tools
from ucl_machine_tools.hosts import HostSpec


def test_remote_python_builder_uses_shared_framed_transport() -> None:
    argv = ssh_tools.build_remote_python_argv("cream", timeout_seconds=8)

    assert argv[0] == "python3"
    assert "--logical-argv" in argv
    assert "cream" in argv
    assert argv[-2:] == ["python3", "-"]


def test_remote_bash_builder_uses_shared_framed_transport() -> None:
    host = HostSpec(name="cream", ssh_host="cream")
    argv = launch.remote_bash_argv(host, "printf hello")

    assert argv[0] == "python3"
    assert "--logical-argv" in argv
    assert "cream" in argv
    assert argv[-5:] == ["bash", "--noprofile", "--norc", "-c", "printf hello"]


def test_virtualbox_startup_block_is_removed_without_hiding_other_errors() -> None:
    text = "\n".join(
        [
            "VBoxManage: Failed to create the VirtualBox object",
            "Document is empty",
            "/home/example-user/.config/VirtualBox/VirtualBox.xml, line 1",
            "NS_ERROR_FAILURE",
            "ssh: actual transport failure",
            "",
        ]
    )

    assert rsync_transport.strip_virtualbox_startup_noise(text.encode()) == b"ssh: actual transport failure\n"


def test_noise_like_command_text_is_unchanged_without_a_virtualbox_block() -> None:
    text = "Document is empty\nNS_ERROR_FAILURE\nactual command output\n"

    assert rsync_transport.strip_virtualbox_startup_noise(text.encode()) == text.encode()


def test_partial_or_unrelated_virtualbox_text_is_never_removed() -> None:
    samples = (
        b"VBoxManage is real command output\n",
        b"VBoxManage: Failed to create the VirtualBox object\nDocument is empty\n",
        b"VirtualBox startup failed for a different reason\n",
    )

    for sample in samples:
        assert rsync_transport.strip_virtualbox_startup_noise(sample) == sample
