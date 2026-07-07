# UCL Machine Tools

Standalone read-only utilities for checking UCL CS GPU availability and
machine-local scratch state.

The inventory command checks:

- GPU availability through `nvidia-smi`
- `/tmp` free space
- whether `/tmp/ucl-machine-tools` exists
- the restart policy recorded for each host class

It does not know about any research project, dataset, model, checkpoint, or run
directory.

## Usage

```bash
scripts/ucl-inventory check barbury-l
scripts/ucl-inventory gpus 3090ti
scripts/ucl-inventory state timeshare
scripts/ucl-inventory recommend barbury-l,canada-l --min-free-vram-gb 20
scripts/ucl-inventory --selector barbury-l --json
```

## TSG Restart Notes

UCL CS TSG says lab PCs regularly reboot on Monday and Thursday evenings between
7:30pm and midnight, and may be rebooted at any time. The timeshare GPU page
lists `blaze`, `cream`, and `vanilla`, but does not list that same lab-PC reboot
window for them.
