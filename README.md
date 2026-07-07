# UCL Machine Tools

Standalone utilities for checking UCL CS GPU availability and launching small
script bundles on remote machines.

The inventory command checks:

- GPU availability through `nvidia-smi`
- `/tmp` free space
- whether `/tmp/ucl-machine-tools` exists
- the restart policy recorded for each host class

It does not know about any research project, dataset, model, checkpoint, or run
directory.

## Inventory

```bash
scripts/ucl-inventory check barbury-l
scripts/ucl-inventory gpus 3090ti
scripts/ucl-inventory state timeshare
scripts/ucl-inventory recommend barbury-l,canada-l --min-free-vram-gb 20
scripts/ucl-inventory --selector barbury-l --json
```

## Launch

`ucl-launch` uploads a local directory with tar over SSH, writes a small remote
launcher, and starts it inside tmux. It always checks/starts the `knuckles` SSH
master connection first.

```bash
scripts/ucl-launch --host barbury-l --local-dir ./bundle --script run.sh
scripts/ucl-launch --host barbury-l --local-dir ./bundle --script run.sh --session work
scripts/ucl-launch --host barbury-l --local-dir ./bundle --script run.sh --new-session
scripts/ucl-launch --host barbury-l --local-dir ./bundle --script run.sh --dry-run
```

Default tmux behavior:

- if exactly one tmux session exists, launch a new window there
- if no tmux sessions exist, create a new session
- if multiple sessions exist, fail and ask for `--session` or `--new-session`

## TSG Restart Notes

UCL CS TSG says lab PCs regularly reboot on Monday and Thursday evenings between
7:30pm and midnight, and may be rebooted at any time. The timeshare GPU page
lists `blaze`, `cream`, and `vanilla`, but does not list that same lab-PC reboot
window for them.
