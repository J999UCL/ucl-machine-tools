# UCL Machine Tools

One command for UCL CS GPU machine checks, small remote commands, and tmux-backed
script launches.

```bash
scripts/ucl status 3090ti
scripts/ucl status barbury-l canada-l
scripts/ucl doctor barbury-l
scripts/ucl exec barbury-l hostname
scripts/ucl exec barbury-l --cwd /tmp --timeout 60 pwd
scripts/ucl exec barbury-l --shell csh --stdin < setup_python.csh
scripts/ucl exec barbury-l --detach --new-session -- hostname
scripts/ucl run --host barbury-l --local-dir ./bundle --script run.sh
scripts/ucl tail last
scripts/ucl fetch last
```

The tool always checks/starts the `knuckles` SSH master connection before remote
work. Remote transfers use SSH/tar streams, not `scp`, `sftp`, or `rsync`.

## Commands

- `ucl status [target]` checks GPU availability, `/tmp` free space,
  `/tmp/ucl-machine-tools`, and restart policy.
- `ucl doctor HOST` checks one host, tmux visibility, and scratch state.
- `ucl exec HOST COMMAND...` runs a short remote command synchronously and
  prints stdout/stderr directly. Use `--cwd DIR`, `--timeout SECONDS`, or
  `--json` when helpful.
- `ucl exec HOST --stdin` runs a stdin script synchronously, avoiding nested
  quote problems for multi-line commands. Use `--shell csh` when the script
  needs to source UCL `.csh` setup files.
- `ucl exec HOST --detach -- COMMAND...` uses the old tmux-backed async path
  and records the run for `tail`/`fetch`.
- `ucl run` uploads a local bundle, writes launcher files, and starts the bundle
  script in tmux.
- `ucl tail last`, `ucl fetch last`, and `ucl clean HOST` operate on recorded
  run metadata.

## Environment Setup

The tool deliberately does not encode Python, PyTorch, uv, conda, or project
setup. Put setup commands in the script you launch, or send them with
`ucl exec --stdin`.

For UCL TSG Python setup:

```bash
scripts/ucl exec barbury-l --shell csh --stdin <<'CSH'
source /opt/Python/Python-3.11.5_Setup.csh
pip install torch --user
CSH
```

For bash-native work, keep the default shell:

```bash
scripts/ucl exec barbury-l df -h /tmp
```

## Tmux Rules

`ucl exec` only uses tmux when `--detach` is passed. Detached exec is
intentionally conservative:

- one existing session: launch a new window there
- zero sessions: fail unless `--session` or `--new-session` is passed
- multiple sessions: fail unless `--session` or `--new-session` is passed

`ucl run` may create a generated session when no tmux session exists, because it
is always the long-job launch path.

## TSG Restart Notes

UCL CS TSG says lab PCs regularly reboot on Monday and Thursday evenings between
7:30pm and midnight, and may be rebooted at any time. The timeshare GPU page
lists `blaze`, `cream`, and `vanilla`, but does not list that same lab-PC reboot
window for them.
