# UCL Machine Tools

One command for UCL CS GPU machine checks, small remote commands, and tmux-backed
script launches.

```bash
scripts/ucl status 3090ti
scripts/ucl status barbury-l canada-l
scripts/ucl doctor barbury-l
scripts/ucl exec barbury-l hostname
scripts/ucl exec barbury-l canada-l -- hostname
scripts/ucl exec 3090ti --gpu auto --min-free-vram-gb 20 -- nvidia-smi
scripts/ucl exec barbury-l --cwd /tmp --timeout 60 pwd
scripts/ucl exec barbury-l --shell csh --stdin < setup_python.csh
scripts/ucl exec barbury-l --detach --new-session -- hostname
scripts/ucl stage --uv --host barbury-l --name demo --local-dir ./project --remote-root /tmp/thakwani/demo
scripts/ucl run --stage demo-barbury-l-STAGE_HASH --script scripts/run.sh --new-session
scripts/ucl run --host barbury-l --new-session --gpu auto --min-free-vram-gb 20 --local-dir ./bundle --script run.sh
scripts/ucl run --host barbury-l --session my_run --remote-root /tmp/ucl-machine-tools/fpt/launchers --local-dir ./bundle --script run.sh
scripts/ucl tail last --live
scripts/ucl fetch last
scripts/ucl jobs
scripts/ucl copy ./data barbury-l:/tmp/ucl-machine-tools/data --verify size
scripts/ucl copy barbury-l:/tmp/checkpoints barnacle-l:/tmp/checkpoints --verify sha256 --reuse-from barnacle-l:/tmp/checkpoints.previous --retries 3
scripts/ucl env barbury-l --remote-root /tmp/ucl-machine-tools/fpt --json
```

The tool always checks/starts the `knuckles` SSH master connection before remote
work. Every remote command uses a nonce-framed transport that discards login-hook
output before the command starts. In particular, broken VirtualBox startup
diagnostics cannot corrupt probes, checksums, rsync, tmux control, or command
results. Once the frame is established, command stdout and stderr are forwarded
unchanged, including command output that happens to mention VirtualBox.
Internal shell helpers use Bash without startup profiles. An explicit
`ucl exec HOST -- bash -lc '...'` is likewise normalized to
`bash --noprofile --norc -c`, preventing a nested login shell from printing
profile hooks such as dates or `nvidia-smi` tables inside the command result.

## Commands

- `ucl status [target]` checks GPU availability, `/tmp` free space,
  `/tmp/ucl-machine-tools`, and restart policy. Human output streams one row as
  soon as each host responds, using up to 32 concurrent probes and a five-second
  per-host handshake timeout by default. `--json` remains buffered and catalog
  ordered for scripts.
- `ucl doctor HOST` checks one host, tmux visibility, and scratch state.
- `ucl exec HOST_OR_SELECTOR [HOST_OR_SELECTOR ...] COMMAND...` runs a short
  remote command synchronously. For multiple hosts, use `--` before the command,
  e.g. `ucl exec barbury-l canada-l -- hostname`.
- `ucl exec HOST --stdin` runs a stdin script synchronously, avoiding nested
  quote problems for multi-line commands. Use `--shell csh` when the script
  needs to source UCL `.csh` setup files.
- `ucl exec HOST --detach -- COMMAND...` uses the old tmux-backed async path
  and records the run for `tail`/`fetch`.
- `ucl stage --uv` validates a locked UV project, uploads an ignore-aware,
  content-addressed source snapshot, and starts exact environment setup in a
  dedicated tmux session. It prints both the reusable stage ID and setup run ID.
- `ucl run` uploads a local bundle, writes launcher files, and starts the bundle
  script in tmux. It requires `--session NAME` or `--new-session`.
- `ucl run --stage STAGE_ID` verifies the remote ready state, source,
  environment, UV binary, Python interpreter, and requested script before
  launch. It runs with `uv run --frozen --no-sync` and performs no upload or
  dependency sync.
- `ucl jobs`, `ucl info`, `ucl stop`, `ucl tail`, `ucl fetch`, and `ucl clean`
  operate on recorded run metadata.
- New detached jobs record the tmux socket, pane, process, and Linux session
  identity. `ucl stop` refuses legacy or mismatched identities, signals only
  that recorded session through pidfds, and never deletes tmux panes/sessions.
- `ucl tail RUN_ID --live` prints the latest lines and streams new output until
  interrupted with `Ctrl-C`; `--follow` remains an alias.
- Recorded `ucl run` and detached `ucl exec` jobs include provenance: project
  tag, local git SHA when available, script hash, bundle path, selected GPU,
  remote root, and env keys with values redacted. Add `--project NAME` when
  you want `ucl jobs` to disambiguate work across projects.
- `ucl copy SRC DST [-- RSYNC_ARGS...]` copies between local paths or
  `HOST:/absolute/path` endpoints; host aliases/selectors must resolve to exactly
  one UCL host. For lab-machine-to-lab-machine copies, rsync runs from the source
  host so data stays inside UCL. Its SSH hops use the same global nonce-framed
  transport: a missing, late, or oversized handshake fails closed and never
  retries through raw SSH.
  Remote copies require `python3` and `rsync` in each participating host's
  non-interactive `PATH`.
  With `--verify sha256`, `ucl copy` pre-compares
  source and destination, skips exact files, transfers missing or mismatched
  files, verifies the result, and retries as needed. `--retries N` controls the
  retry count. `--reuse-from HOST:/absolute/path` enables same-filesystem
  hard-link reuse from an existing copy when content, mode, and mtime match.
  Verified directory copies treat `SRC` and `DST` as tree roots, so the contents
  of `SRC` map directly into `DST`; `DST` must not contain source-absent files.
  `--partial` keeps resumable data under `DST/.ucl-rsync-partial`. Raw rsync
  arguments remain available for unverified copies only. Remote arguments use
  rsync's protected-argument mode. `-e`, `--rsh`, `--rsync-path`, remote-side
  options (`-M`/`--remote-option`), legacy argument mode, and options that
  disable protected arguments are rejected. Use `--verify size` when byte-level
  identity is not required.
- `ucl env HOST --remote-root DIR` checks reachability, scratch/root state, TSG
  setup scripts, `/tmp` space, and optional GPU availability.
- `--gpu auto` on `exec`, `run`, and `env` requires 20 GB free VRAM by default;
  tune it with `--min-free-vram-gb`. Existing GPU processes and utilization do
  not veto launch selection; a GPU is eligible when it meets the VRAM threshold.

## Launcher Root

Detached `ucl exec`, `ucl run`, and `ucl clean` default to
`/tmp/ucl-machine-tools/launchers`. Override that with `--remote-root DIR` or set
`UCL_LAUNCH_ROOT`, for example `/tmp/ucl-machine-tools/fpt/launchers`. Explicit
`--remote-dir` values must stay under the selected root.

## Automatic UV Environments

Projects staged with `ucl stage --uv` must contain `pyproject.toml`, `uv.lock`,
and `.python-version`. The local lock must pass `uv lock --check`; staging never
updates it. Commit those three files with the project and use `uv lock`
intentionally when dependencies change.

The source snapshot follows nested `.gitignore` and `.uclignore` files and
always excludes Git metadata, virtual environments, caches, secrets such as
`.env`, and top-level data/output/checkpoint directories. The exact selected
files are copied into a stable local snapshot, rehashed, verified remotely, and
atomically promoted into
`REMOTE_ROOT/stages/NAME/sources/SHA256`. A changed included file creates a new
stage identity; ignored data does not.

Remote setup sources `/opt/Python/Python-3.11.5_Setup.csh`, installs the exact
local UV release through Astral's versioned standalone installer, keeps UV,
Python downloads, package cache, source, and environment under the requested
remote root, and runs `uv sync --frozen --no-editable` followed by
`uv sync --frozen --check`. UV may use the TSG Python or install the version
requested by `.python-version`. Concurrent setup for the same UV tool or
environment is lock-serialized.

The stage identity includes the remote root and a digest of setup environment
values, so build-affecting changes cannot silently reuse another environment.
Environment values themselves are not stored in the stage registry.

Setup is asynchronous. Follow it with `ucl tail SETUP_RUN_ID --live`; once its
state is ready, launch any script already in the snapshot with:

```bash
scripts/ucl run --stage STAGE_ID --script scripts/run.sh --new-session
```

The local `scripts/ucl` wrapper bootstraps its own small staging dependencies
through the repository's `uv.lock` only when `ucl stage` needs them. It does not
install anything into the system Python.

## Tmux Rules

`ucl exec` only uses tmux when `--detach` is passed. Detached exec is
intentionally conservative:

- one existing session: launch a new window there
- zero sessions: fail unless `--session` or `--new-session` is passed
- multiple sessions: fail unless `--session` or `--new-session` is passed

`ucl run` is also explicit: pass `--session NAME` to use/create a named session,
or `--new-session` to create a generated session. It never silently reuses the
only existing tmux session.

## TSG Restart Notes

UCL CS TSG says lab PCs regularly reboot on Monday and Thursday evenings between
7:30pm and midnight, and may be rebooted at any time. The timeshare GPU page
lists `blaze`, `cream`, and `vanilla`, but does not list that same lab-PC reboot
window for them. On Monday and Thursday, `ucl status` shows a Europe/London
countdown for reachable lab PCs. During the active window its restart column
labels unreachable lab PCs as restarting and warns that reachable hosts may
shut down at any time.
