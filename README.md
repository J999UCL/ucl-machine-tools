# UCL Machine Tools

One command for UCL CS GPU machine checks, small remote commands, and tmux-backed
script launches.

```bash
scripts/ucl status 3090ti
scripts/ucl doctor barbury-l --profile tsg-pytorch --gpu 0
scripts/ucl exec barbury-l -- hostname
scripts/ucl exec barbury-l --profile tsg-pytorch -- python3 -c 'import torch; print(torch.cuda.is_available())'
scripts/ucl run --host barbury-l --local-dir ./bundle --script run.sh
scripts/ucl tail last
scripts/ucl fetch last
```

The tool always checks/starts the `knuckles` SSH master connection before remote
work. Remote transfers use SSH/tar streams, not `scp`, `sftp`, or `rsync`.

## Commands

- `ucl status [target]` checks GPU availability, `/tmp` free space,
  `/tmp/ucl-machine-tools`, and restart policy.
- `ucl doctor HOST` checks one host, tmux visibility, scratch state, and an
  optional launch profile.
- `ucl exec HOST -- COMMAND...` writes a tiny remote launcher and starts it in
  tmux. It reuses the single existing tmux session by default. If zero or
  multiple sessions exist, pass `--session NAME` or `--new-session`.
- `ucl exec HOST --stdin` reads a bash script from stdin, avoiding nested quote
  problems for multi-line commands.
- `ucl run` uploads a local bundle, writes profile-aware launcher files, and
  starts the bundle script in tmux.
- `ucl tail last`, `ucl fetch last`, and `ucl clean HOST` operate on recorded
  run metadata.

## Profiles

Profiles are JSON and dependency-free. Load order is:

```text
configs/launch_profiles.json
~/.config/ucl-machine-tools/launch_profiles.json
--profile-file PATH
CLI --env values
```

Built-ins:

- `plain-bash`: no setup.
- `uv`: require `uv`, then run commands through `uv run --`.
- `tsg-pytorch`: source TSG Python/CUDA setup and require torch CUDA.

Profiles may use `extends`, `env`, `source`, `preflight`,
`preflight_after_setup`, and `run_prefix`. Project-specific profiles should live
in user or explicit profile files, not in this generic repo.

## Tmux Rules

`ucl exec` is intentionally conservative:

- one existing session: launch a new window there
- zero sessions: fail unless `--session` or `--new-session` is passed
- multiple sessions: fail unless `--session` or `--new-session` is passed

`ucl run` may create a generated session when no tmux session exists, because it
is the long-job launch path.

## TSG Restart Notes

UCL CS TSG says lab PCs regularly reboot on Monday and Thursday evenings between
7:30pm and midnight, and may be rebooted at any time. The timeshare GPU page
lists `blaze`, `cream`, and `vanilla`, but does not list that same lab-PC reboot
window for them.
