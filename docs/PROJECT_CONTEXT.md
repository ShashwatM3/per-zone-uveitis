# Per-Zone Uveitis Classification — Full Project Context

**Last updated:** 2026-05-28  
**Repository:** [ShashwatM3/per-zone-uveitis](https://github.com/ShashwatM3/per-zone-uveitis)  
**Active branch:** `feature/clip-convnext` (also: `feature/binary-zone-classification`, `main`)  
**Cluster:** LARA156 (4× GPU: 2× RTX A6000, 2× RTX 3090)  
**Python env:** `conda` env `venv` at `/home/shashwat/miniconda3/envs/venv/bin/python` — **not** the project-local `.venv` (CUDA mismatch → CPU fallback)

This document is the single “extreme context” reference: what the project is, how the pipeline works, what experiments were run, what worked, what failed, and where everything lives on disk.

---

## 1. Mission and success criteria

### Clinical / scientific goal

Classify **per-retinal-zone** uveitis severity on **ultra-widefield fundus (UWFA FP)** images. Each fundus image is divided into **10 anatomical zones** (inner/ring × nasal/temporal × upper/lower, plus optic-nerve-head zone). The model predicts whether a zone is **healthy (tier 0)** vs **diseased (tiers 1+2 merged)** at the zone level.

This is **not** a whole-image diagnosis task. One patient visit yields up to 10 correlated zone samples; evaluation must respect **patient-level splits** to avoid leakage.

### Protocol stop conditions (autonomous research loop)

| Criterion | Target | Status (2026-05-28) |
|-----------|--------|------------------------|
| Test **macro-F1** on canonical split | ≥ **0.70** | **0.5599** (best) |
| Test **class-1 recall** | ≥ **0.65** | **0.5602** (best) |
| Ceiling rule | 25+ protocol iterations with no **>0.02** F1 gain in last 8 runs | 20 completed `protocol_r*` runs; ceiling not yet auto-written |

**Current best run:** `protocol_r14_convnext_exclude_tier1` → test macro-F1 **0.5599**, class-1 recall **0.5602**, best epoch **10**.

Checkpoint: `runs/protocol_r14_convnext_exclude_tier1/best.pt` (~108 MB)

---

## 2. Problem formulation

### Label tiers (spreadsheet semantics)

Annotators assign each zone a severity tier:

| Raw `Zone_Label` | Meaning | Binary mapping |
|------------------|---------|----------------|
| **0** | Healthy / no active uveitis in zone | Class **0** |
| **1** | Ambiguous / borderline (“tier-1”) | Class **1** (merged with tier 2) |
| **2** | Active uveitis in zone | Class **1** |

Binary training collapses `{1, 2} → 1`. The ambiguous tier-1 zones are a major source of label noise and drove several experimental branches.

### Soft-label alternative (not successful here)

`zone_dataset.py` defines soft targets for multiclass CSV:

- Tier 0 → `[1.0, 0.0]`
- Tier 1 → `[0.5, 0.5]` (ambiguous)
- Tier 2 → `[0.0, 1.0]`

Used with `--soft-labels --loss soft_ce` on the multiclass CSV. **Regressed** vs focal baseline (runs r01, r12).

### Exclude-tier-1 (winning strategy)

`--exclude-tier1` removes tier-1 rows from **training only**. Val and test sets are unchanged so comparisons stay fair. Hypothesis: training on ambiguous zones teaches the model the wrong decision boundary. This yielded the largest gain in the protocol (r11 → r14).

### Zone-conditioning

Each sample includes a **zone number** (1–10). The model concatenates a learned **zone embedding** (`--zone-embed-dim 64`) with image features before a 2-layer MLP head (`--head-hidden 256`). This tells the classifier which anatomical region it is looking at.

### Task scale

From `processed_image_arrays_multiclass/summary.json` (multiclass preprocessing):

- **772** matched fundus images → **7,720** zone rows in `zone_training_table.csv`
- **~110** unique patients in the canonical split
- Zone-level rows are **correlated** within a patient (not i.i.d.)

Canonical split zone counts (from r14 `metrics.json`):

| Split | Zone rows | Patients | Class 0 | Class 1 |
|-------|-----------|----------|---------|---------|
| Train | 3,046 | 62 | 2,056 | 990 |
| Val | 750 | 14 | 412 | 338 |
| Test | 840 | 14 | 508 | 332 |

---

## 3. Data pipeline (raw → training tensors)

### Directory layout

```
data/                              # Raw PatientXXX/VisitDate/ fundus images + xlsx annotations
processed_image_arrays/            # Legacy binary pipeline (~58 GB) — pre-cropped zone PNGs
processed_image_arrays_multiclass/ # Current protocol CSV + cleaned/ on-the-fly crops (~556K metadata; images via cleaned/)
splits/canonical_split.json        # Fixed patient IDs for train/val/test (seed=13)
runs/                              # Checkpoints, metrics.json, train.log per experiment
weights/                           # Symlink to RETFound checkpoint
RETFound_mae_natureCFP/            # ~3.7 GB MAE weights (fundus foundation model)
external/RETFound/                 # Vendored official RETFound code
```

Annotations spreadsheet: `data/UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx`

### Stage 1 — Zone extraction (`extract_zones.py`)

Self-contained port of “Slice Zone Mohammad 5.1.2026” logic:

- Auto-detect fovea/crosshair from yellow overlay (OpenCV)
- Remove yellow markup
- Extract **10 zones** as RGBA crops using polar geometry (inner radius 3 mm, outer 16 mm, ONH offset)
- Zone names: inner/ring × nasal/temporal × upper/lower + ONH

### Stage 2 — Preprocessing (`pre_processing.py`)

- Walks `data/Patient*/`
- Standardizes images (e.g. flip OS eyes to OD convention)
- Caches cleaned `.npy` + geometry `.json` sidecars
- Builds `zone_training_table.csv`
- `--multiclass-zone-labels` keeps raw 0/1/2 in CSV (required for exclude-tier1 and soft labels)

### Stage 3 — Dataset (`zone_dataset.py`)

- **Patient-level split** via `load_or_create_split()` — same patient never in two splits
- `splits/canonical_split.json` is **frozen** for all protocol runs (reproducibility)
- Supports legacy pre-cropped PNGs (`Zone_Image` column) and new on-the-fly crops (`Cleaned_Image` + JSON geometry)
- Optional sample weights: tier-confidence down-weighting, balanced batch sampler

---

## 4. Models and training scripts

### 4.1 ConvNeXt-Tiny — primary workhorse (`train_convnext.py`)

| Setting | Best (r14) value |
|---------|------------------|
| Backbone | `convnext_tiny` (ImageNet pretrained, timm) |
| Image size | 288×288 |
| Loss | Focal, γ=**3.0** |
| Class weighting | **inverse** frequency |
| LR | 1e-4 head, 1e-5 backbone (0.1×) |
| Scheduler | Cosine |
| Batch | 32 |
| Epochs | 20 (early stop @ 17; best @ 10) |
| Patience | 8 on val macro-F1 |
| AMP | On |
| Zone embed | 64 |
| Exclude tier-1 | **Yes** |
| CSV / root | `processed_image_arrays_multiclass/` |

**Also supports:** EfficientNet-B3 (`--backbone efficientnet_b3`), MixUp, balanced batches (BBFL), effective number class weights, positive oversampling, tier-confidence weights, soft CE.

**Selection metric:** validation **macro-F1** → saves `best.pt` → loads for test eval → writes `metrics.json` + W&B.

### 4.2 OpenCLIP ConvNeXt-Large-D (`train_clip_convnext.py`)

- `convnext_large_d_320` + LAION weights via `open_clip`
- Checkpoints ~**764 MB** each (full CLIP in `best.pt`)
- Explored in protocol r02–r06; **never beat** ConvNeXt-Tiny on canonical test split
- Baseline before protocol: `clip_convnext_10ep_baseline` F1=**0.5251**

### 4.3 RETFound MAE ViT (`train_retfound.py`)

- Fundus foundation model ([Zhou et al., Nature 2023](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP))
- Vendored code: `external/RETFound/`
- Weights: `weights/RETFound_mae_natureCFP.pth` → `RETFound_mae_natureCFP/` (~3.95 GB)
- Zone-aware head (same pattern as ConvNeXt)
- Freeze backbone 4–5 epochs, then unfreeze at `--backbone-lr`
- Image size **224** (ViT standard)
- **Underperformed** ConvNeXt r14 on this task despite longer “correct recipe” run

### 4.4 Losses (`losses.py`)

- Weighted **cross-entropy**
- **Focal loss** (γ configurable) with class alpha from inverse/effective weighting
- **Soft CE** for soft tier targets

### 4.5 Post-hoc evaluation (no retrain)

- `eval_convnext_sweep.py` — threshold sweep + 4-view flip TTA
- `threshold_sweep.py`, `tta_sweep.py` — standalone utilities

**Finding:** On r14 checkpoint, optimal threshold stayed **0.50**; TTA **hurt** (0.5483 vs 0.5599).

---

## 5. Evaluation metrics

All training scripts report:

- Accuracy, **balanced accuracy**, **macro-F1** (primary)
- Per-class precision, recall, F1, specificity
- 2×2 confusion matrix
- W&B logs per-epoch train/val curves + test ROC/AUC (ConvNeXt)

**Why macro-F1:** Zone labels are imbalanced (~62/38 in test); macro-F1 treats both classes equally.

**Why patient split:** 10 zones per visit are correlated; random zone splits would inflate performance.

---

## 6. Canonical split

**File:** `splits/canonical_split.json`  
**Seed:** 13 (used when split was first created from `clip_convnext_10ep_baseline` era)

- **62** train patients, **14** val, **14** test
- Every `protocol_r*` run uses this file via `--split-json splits/canonical_split.json`
- `metrics.json` in each run embeds the same split for auditability

---

## 7. Research protocol — structure and branches

The May 2026 research loop on `feature/clip-convnext` followed a **hypothesis-driven protocol**: one major change per run, compare to best, log in `RESEARCH_LOG.md`.

### Experimental branches (conceptual)

| Branch | Idea | Outcome |
|--------|------|---------|
| **A** — Backbone swap | CLIP-L, EfficientNet-B3, RETFound | CLIP ≈ baseline; EfficientNet NaN collapse; RETFound below ConvNeXt |
| **B** — Imbalance / loss | focal γ, oversample, remove CW, BBFL, balanced batches | γ=3 + inverse CW best; oversample/MixUp/CW-none hurt |
| **C** — Label handling | Soft labels, tier-confidence weights, **exclude tier-1** | Exclude tier-1 **won**; soft labels and tier weights failed |
| **D** — Inference tricks | Threshold sweep, TTA, MixUp training | No free lunch at inference; MixUp hurt training |

### Autonomous orchestration scripts

| Script | Purpose |
|--------|---------|
| `run_protocol_batch.py` | Sequential protocol runner + `RESEARCH_LOG.md` git commits |
| `research_loop.sh` | Early research loop wrapper |
| `protocol_autonomous.sh` | Full autonomous queue |
| `protocol_autonomous_phase3.sh` | r19 RETFound + r22–r25 one-change sweeps |
| `run_autonomous_loop.sh` | Watcher-friendly loop |
| `run_autonomous_resume.sh` | r26–r28, sweep, 5-fold CV, ceiling check |
| `run_autonomous_watcher.sh` | Restarts loop on failure |
| `scripts/check_protocol_ceiling.py` | Writes `CEILING_REPORT.md` when rules trigger |
| `scripts/monitor_r26_retfound.sh` | Polling monitor for long RETFound job |

**Operational notes:**

- GPU **0** pinned via `CUDA_VISIBLE_DEVICES=0` in autonomous scripts
- `free_disk()` in resume script deletes `wandb/run-*` and stray `checkpoint_epoch*.pt`
- Phase 3 had a **typo** (`--wandb-run-name ..."`) that crashed once; fixed
- User issued **stop** — all autonomous training killed; r26 ConvNeXt g35 never finished
- `run_autonomous_loop.sh` nesting bug: resume block (CV, r26+) did not chain when only phase3 invoked

### Cross-validation

**Completed:** 3-fold patient CV with r14 config (`run_patient_cv.py`, `run_cv3_r14.sh`)

| Fold | Test F1 | Class-1 recall |
|------|---------|----------------|
| 0 | 0.5323 | 0.4460 |
| 1 | 0.5223 | 0.3382 |
| 2 | 0.5032 | 0.4208 |
| **Mean ± std** | **0.519 ± 0.012** | **0.402 ± 0.046** |

**Interpretation:** Canonical single-split r14 (0.5599) is **optimistic** vs CV mean (~0.52). Fixed split may be slightly favorable; still far from 0.70 goal.

**Not completed:** 5-fold CV (`runs/protocol_cv5_r14_config/` stub only)

---

## 8. Complete experiment chronology

### Pre-protocol baselines (branch exploration)

| Run | Backbone | Test F1 | Notes |
|-----|----------|---------|-------|
| `clip_convnext_10ep_baseline` | CLIP ConvNeXt-L | 0.5251 | Starting best before protocol |
| Various `convnext_*`, `clip_*` smoke runs | Mixed | — | Early architecture comparison |

### Protocol runs r01–r28+ (canonical split)

Sorted by test macro-F1:

| Run | Test F1 | Cls-1 recall | Best ep | One-line summary |
|-----|---------|--------------|---------|------------------|
| **r14** `exclude_tier1` | **0.5599** | **0.5602** | 10 | **BEST** — focal γ=3, inverse CW, exclude tier-1, 20ep |
| r11 `focal_g3` | 0.5461 | 0.5271 | 9 | Previous best; tier-1 still in training |
| r20 `14ep` (r14 config) | 0.5368 | 0.4669 | 14 | Shorter training hurts vs r14 |
| r22 `γ=2.5` | 0.5345 | 0.4578 | 14 | Lower gamma worse |
| r25 `balanced_batches` | 0.5309 | 0.6777 | 10 | High recall, lower F1 (precision tradeoff) |
| r18 EfficientNet-B3 | 0.5289 | 0.6175 | 1 | NaN after ep1; unstable |
| r03 ConvNeXt focal | 0.5285 | 0.5000 | 20 | First ConvNeXt beat CLIP |
| r02 CLIP default | 0.5259 | 0.4548 | 17 | Marginal over baseline |
| r17 tier-conf weights | 0.5095 | 0.3614 | 7 | Down-weight tier-1 in loss — failed |
| r16 BBFL | 0.5083 | 0.3946 | 9 | Balanced batches + γ=1.5 — failed |
| r05 CLIP freeze3 | 0.5044 | 0.4819 | 12 | CLIP unfreeze schedule — failed |
| r21 MixUp α=0.2 | 0.5025 | 0.4006 | 11 | Augmentation hurt |
| r04 CLIP focal | 0.4990 | 0.6898 | 16 | High recall, poor F1 |
| r24 zone_embed=128 | 0.4934 | 0.3313 | 13 | Larger zone embedding — failed |
| r19 RETFound 14ep | 0.4889 | 0.5512 | 6 | Fundus MAE < ConvNeXt |
| r23 effective CW | 0.4878 | 0.2319 | 10 | Effective number weighting — failed |
| r01 soft CE | 0.4849 | 0.2681 | 1 | Soft labels — failed |
| r12 softlabels | 0.4849 | 0.2681 | 1 | Same as r01 |
| r15 CW none | 0.4719 | 0.1416 | 9 | Removing inverse weights — collapsed cls-1 |
| **r26** RETFound “correct recipe” | **0.4477** | **0.2319** | 12 | 50ep, freeze 5, lr 1e-3/5e-5, 224px — **worst RETFound** |

### Incomplete / killed runs

| Run | Status |
|-----|--------|
| r13 `focal_g3_os2` | Manually killed @ ep9 — class-1 overprediction (recall 0.94) |
| r26 `convnext_excl_t1_g35` | Killed before epoch 1 — only `train.log` + W&B config |
| r27, r28 | Never started (autonomous stopped) |

### r26 RETFound “correct recipe” (user-requested)

Deliberate long run separate from protocol 14ep recipe:

```
50 epochs, batch 16, image 224, focal γ=3, inverse CW, exclude-tier1
lr=1e-3 head, backbone-lr=5e-5, freeze 5 epochs, patience 15, WD=0.05
```

- Early stop @ epoch **27** (no val improvement 15 epochs after best @ **12**)
- Best val F1 **0.5806** but test F1 **0.4477** — severe **val/test gap** / overfitting after backbone unfreeze
- W&B: [1jb9295h](https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone/runs/1jb9295h)
- Artifacts: `runs/protocol_r26_retfound_correct_recipe/`

---

## 9. Best configuration (copy-paste ready)

```bash
export CUDA_VISIBLE_DEVICES=0
conda activate venv  # /home/shashwat/miniconda3/envs/venv

python train_convnext.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --split-json splits/canonical_split.json \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 \
  --epochs 20 --batch-size 32 --num-workers 8 --image-size 288 \
  --zone-embed-dim 64 --head-hidden 256 \
  --output-dir runs/protocol_r14_convnext_exclude_tier1 \
  --wandb-project uveitis-per-zone \
  --wandb-run-name protocol_r14_repro
```

**r14 per-class test:** class-0 F1=0.614 recall=0.571; class-1 F1=0.505 recall=0.560.

---

## 10. Key findings and diagnosis

1. **ConvNeXt-Tiny beats CLIP-L and RETFound** on this zone task at current data scale — simpler model + 288px crops + ImageNet pretrain is sufficient; foundation models did not help.

2. **Exclude ambiguous tier-1 from training** is the single largest validated improvement — suggests label noise in tier-1 dominates the decision boundary.

3. **Stacking imbalance tricks fails** — oversampling + focal + inverse weights together (r13) caused collapse; removing weights (r15) also collapsed class-1.

4. **focal γ=3** sweet spot — γ=2.5 (r22) and BBFL γ=1.5 (r16) both worse than γ=3.

5. **Inference tricks don’t rescue a weak checkpoint** — threshold 0.50 optimal; TTA degrades.

6. **3-fold CV ~0.52** vs canonical **0.56** — report both when claiming generalization; 0.70 goal likely needs more data, better labels, or architectural change (e.g. MIL over zones).

7. **Disk pressure** — full disk (100%) caused write failures; CLIP checkpoints (~764 MB×8 GB) and RETFound checkpoints (~1.2 GB×2) are main reclaimable training artifacts.

---

## 11. Logging and artifact conventions

### Per-run outputs (`runs/<name>/`)

| File | Contents |
|------|----------|
| `metrics.json` | Full args, split, data summary, `history[]`, `best_val`, `test` |
| `best.pt` | `model_state`, `args`, `split`, `epoch`, `val_metrics` |
| `train.log` | Stdout including per-epoch val lines |

### Project-level logs

| File | Contents |
|------|----------|
| `RESEARCH_LOG.md` | Human narrative per iteration (hypothesis, diagnosis, next) |
| `runs/PROTOCOL_STATUS.md` | Autonomous queue snapshot |
| `runs/autonomous_loop.log` | Timestamped orchestrator log |
| `WANDB_RUNS.md` | W&B URL index after local `wandb/run-*` prune |
| `docs/RETFOUND_SETUP.md` | HF gated weights setup |

### Weights & Biases

- Project: `uveitis-per-zone` on [wandb.ai/smahalanobis-uc-davis](https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone)
- Local `wandb/run-*` pruned 2026-05-28; cloud runs retained
- RETFound logs only final test scalars; ConvNeXt logs full epoch curves

---

## 12. Repository map (code)

```
extract_zones.py          # 10-zone geometry + yellow removal
pre_processing.py         # data/ → NPZ + zone_training_table.csv
zone_dataset.py           # ZoneRecord, patient split, ZoneImageDataset
losses.py                 # CE, focal, soft CE
train_convnext.py         # Main trainer (ConvNeXt / EfficientNet)
train_clip_convnext.py    # OpenCLIP ConvNeXt-Large-D
train_retfound.py         # RETFound MAE ViT fine-tune
eval_convnext_sweep.py    # Threshold + TTA on saved checkpoint
run_patient_cv.py         # K-fold patient CV driver
tier1_audit.py            # Tier-1 label analysis utility
tests/check_zone_labels.py
splits/canonical_split.json
scripts/check_protocol_ceiling.py
scripts/monitor_r26_retfound.sh
```

---

## 13. Git history (high level)

Recent commits on `feature/clip-convnext`:

- `ee22b6f` — FocalLoss sample weighting, tier-confidence, backbone args, 14ep default, RETFound fixes
- `bd7ba40` — Positive oversampling, research log iterations 11–13
- `cc604ca` — r03 ConvNeXt focal first protocol improvement
- Protocol results committed via `run_protocol_batch.py` (`result:` / `pre-run:` messages)

---

## 14. Current state (2026-05-28)

| Item | Status |
|------|--------|
| Training jobs | **None running** |
| Best model | `protocol_r14` @ 0.5599 |
| Protocol ceiling report | **Not written** (conditions not fully met) |
| 5-fold CV | **Not run** |
| r27, r28, r26 ConvNeXt g35 | **Not completed** |
| Autonomous loop | **Stopped** by user |
| Disk | Was **100% full**; `wandb/run-*` pruned; `temp/ingest_smoke` removed to free ~1 GB |
| Branch | `feature/clip-convnext` |

---

## 15. Disk usage guide (what’s safe to delete)

| Path | ~Size | Deletable? |
|------|-------|------------|
| `runs/clip_*`, `runs/protocol_r0[2456]*` (CLIP) | ~8 GB | **Yes** if metrics saved — huge CLIP checkpoints |
| `runs/protocol_r19_*`, `runs/protocol_r26_retfound_*` | ~2.4 GB | **Maybe** — RETFound 1.2 GB ckpts each |
| `runs/protocol_r{01..25}_*` except **r14** | ~2 GB | **Maybe** — keep r14 `best.pt` |
| `processed_image_arrays/` | 58 GB | **Risky** — verify multiclass `cleaned/` is self-contained |
| `data/` | 10 GB | Only if raw data backed up elsewhere |
| `RETFound_mae_natureCFP/` | 3.7 GB | **No** — needed for RETFound runs |
| `wandb/run-*` | — | **Already pruned** |

---

## 16. Related documents

- `README.md` — Quickstart, env setup, basic train commands
- `RESEARCH_LOG.md` — Iteration-by-iteration narrative (iter 11–28)
- `WANDB_RUNS.md` — W&B run ID → local path mapping
- `docs/RETFOUND_SETUP.md` — HuggingFace gated weights
- `runs/protocol_cv3_r14_config/cv_summary.json` — CV numbers
- `runs/PROTOCOL_STATUS.md` — Last autonomous queue state

---

## 17. Open questions / plausible next steps

These are **not scheduled** — documented for continuity:

1. **Data:** More patients, tier-1 relabeling, or excluding tier-1 from eval as sensitivity analysis
2. **Model:** Patient-level MIL over 10 zones; per-zone specialists; stronger augmentation
3. **Training:** Retry r14 at 20ep with fresh seed CV; ensemble r11+r14
4. **Infrastructure:** Finish 5-fold CV; run `check_protocol_ceiling.py` after 5 more iterations
5. **EfficientNet:** Fix NaN (lower LR, no AMP) if revisiting branch A3
6. **RETFound:** Likely deprioritized unless preprocessing moves to 224px native fundus crops with different head

---

*This file is meant to onboard a new collaborator or future agent with zero prior chat context. For live metrics, always prefer `runs/<name>/metrics.json` over this document.*
