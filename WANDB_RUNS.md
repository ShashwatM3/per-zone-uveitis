# W&B local run index (pruned 2026-05-28)

Local `wandb/run-*` folders were removed after recording this index. Charts and metrics remain on the [uveitis-per-zone](https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone) project.

## What is preserved elsewhere

| Source | Location | Contents |
|--------|----------|----------|
| Training scripts | `train_convnext.py`, `train_retfound.py`, `train_clip_convnext.py` | W&B optional; completed runs write `runs/<name>/metrics.json` |
| Per-epoch text log | `runs/<name>/train.log` | Epoch lines (`val_macro_f1`, per-class) |
| Checkpoints | `runs/<name>/best.pt` | Best val checkpoint + embedded args/split |
| Protocol summary | `RESEARCH_LOG.md` | Completed protocol runs (test F1, hypotheses) — no W&B URLs |
| CV | `runs/protocol_cv3_r14_config/cv_summary.json` | 3-fold summary (`--no-wandb`) |

## Per-run mapping (local `wandb/run-*` at prune time)

### `run-20260528_164709-1jb9295h` — fully mirrored locally

| Field | Value |
|-------|-------|
| W&B name | `protocol_r26_retfound_correct_recipe` |
| W&B URL | https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone/runs/1jb9295h |
| Output dir | `runs/protocol_r26_retfound_correct_recipe/` |
| `metrics.json` | Yes — full `args`, `split`, `history[]`, `best_val`, `test` |
| `train.log` | Yes |
| `best.pt` | Yes (epoch 12) |
| Git commit | `ee22b6f` |
| W&B-only metrics | `test/macro_f1`, `test/recall_class_1` (subset of `metrics.json`) |
| Test macro-F1 | **0.4477** |
| Test class-1 recall | **0.2319** |

`train_retfound.py` does not log per-epoch metrics to W&B; nothing experiment-critical was lost.

### `run-20260528_143426-18eibx8n` — incomplete run

| Field | Value |
|-------|-------|
| W&B name | `protocol_r26_convnext_excl_t1_g35` |
| W&B URL | https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone/runs/18eibx8n |
| Output dir | `runs/protocol_r26_convnext_excl_t1_g35/` |
| `metrics.json` | **No** (killed before epoch 1) |
| `train.log` | Yes — W&B banner + config only |
| `best.pt` | **No** |
| Git commit | `12ac4c6` |
| Notes | ConvNeXt r14-style + `focal_gamma=3.5`, 14 ep; no test metrics |

Per-epoch W&B curves (if any) are **only on the cloud** — see URL above. CLI/config also in `runs/protocol_r26_convnext_excl_t1_g35/train.log`.

## Not duplicated in repo (minor)

- Pip freeze at run start (`requirements.txt`) — W&B cloud only
- Host/GPU metadata — diagnostic only
- W&B client `debug*.log` — not experiment results

## Re-find W&B URL

Search `train.log` for `View run at https://wandb.ai/.../runs/<id>`.
