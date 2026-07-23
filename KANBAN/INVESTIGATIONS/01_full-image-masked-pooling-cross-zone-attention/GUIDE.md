# GUIDE — How to run the Option A experiment (end to end)

**Investigation:** `01_full-image-masked-pooling-cross-zone-attention`
**Read `DESCRIPTION.md` in this folder first.** This file is the operational runbook: environment,
data, the exact commands, what to log, and how to judge the result. It is written so an agent (or a
human) can execute it without re-deriving context.

> **Golden rule for this project:** the only number that counts is **5-fold patient CV vs the
> re-baselined CV baseline** (`DESCRIPTION.md §5, §8`). Single-split test F1 has produced false
> signals three times. Never claim a win from a single split.

---

## 0. TL;DR sequence

1. Confirm environment + GPU (§1).
2. Land the **Phase 0 foundations** — `eval_convnext_sweep.py:116` fix, fovea-fallback audit +
   sidecar reprocess, new 156-patient split, 5-fold CV harness (§3). These are prerequisites, not
   optional.
3. Re-baseline the **existing crop ConvNeXt** under 5-fold CV on the new split (§4). This is the
   denominator.
4. Implement + train **Option A** (`train_zone_attention.py`) under the same 5-fold CV (§5–§6).
5. Compare CV-to-CV, check the success criterion, write results (§7–§8).

---

## 1. Environment

- **Python / conda env:** use the project venv interpreter directly (do not rely on `conda activate`):
  ```
  /home/shashwat/miniconda3/envs/venv/bin/python
  ```
  Referred to below as `$PY`. Set it once per shell:
  ```bash
  cd /home/shashwat/per-zone-uveitis
  PY=/home/shashwat/miniconda3/envs/venv/bin/python
  ```
- **Dependencies** are already installed in that env (`requirements.txt`: torch, torchvision, timm,
  open_clip_torch, numpy, scikit-learn, pillow, opencv-python, openpyxl, **wandb 0.26.1**). If you add
  a new import, install it into this env, not base.
- **GPU:** this box has 4 GPUs — 2× RTX A6000 (49 GB) and 2× RTX 3090 (24 GB). Pick a free one with
  `nvidia-smi` and pin it. **Prefer an A6000 (GPU 0 or 3) for Option A**, because higher-resolution
  input (≈512px, per `DESCRIPTION.md §3.3`) and full-image batches use more memory than the crop
  pipeline. Example:
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  export MKL_THREADING_LAYER=GNU   # numpy/MKL threading; used by the CV runner
  ```
- **W&B:** logging is on by default to project **`uveitis-per-zone`**
  (`https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone`); credentials are in `~/.netrc`. Pass
  `--no-wandb` to disable. Keep W&B **on** for real experiments — it is the primary experiment logger.

---

## 2. Data & key files (know these before you touch anything)

- **Labels CSV (multiclass, keeps raw 0/1/2):**
  `processed_image_arrays_multiclass/zone_training_table.csv`
  **Data root:** `processed_image_arrays_multiclass`
  Use the **multiclass** CSV/root — it preserves raw tiers, which `--exclude-tier1`, soft labels, and
  tier-confidence weighting all require. (A binary-only `processed_image_arrays/` also exists; don't
  use it for this work.)
- **Current frozen split (stale — do NOT reuse for this investigation):**
  `splits/canonical_split.json` (seed 13, 91 patients). Phase 0 replaces this.
- **Zone geometry / masks:** `extract_zones.py` — `make_zone_mask`, `make_masks`, and constants
  (`PX_PER_MM=53`, ONH offset). This is where the **feature-resolution masks** and the
  **geometry-based positional encoding** (`DESCRIPTION.md §3.3`) come from. Reuse `make_zone_mask`;
  do **not** use `apply_zone_and_crop` (that is the crop path being removed).
- **Dataset:** `zone_dataset.py` — `load_zone_records`, `ZoneImageDataset`, `load_or_create_split`,
  `records_for_patients`, `make_patient_split`. Option A needs a **new** dataset item that yields
  `(full_image, 10 feature-resolution zone masks, 10 labels, zone geometry)` per patient-eye instead
  of one crop per row.
- **Current trainer (crop baseline, keep it runnable):** `train_convnext.py`. Reuse its wandb wiring,
  arg conventions, optimizer/schedule/AMP/early-stop, `losses.build_loss`, and metric code as
  reference for the new trainer.
- **CV harness:** `run_patient_cv.py` (currently 3-fold, launches `train_convnext.py` per fold with
  `--no-wandb`). Extend to 5-fold and to the new trainer.
- **Fovea fallback flag:** written by `pre_processing.py` into each image's sidecar `.json`
  (`fovea_fallback=True` when the crosshair wasn't detected).

---

## 3. Phase 0 — prerequisites (must land before Option A results count)

See `DESCRIPTION.md §5` for the rationale. Concretely:

### 3.1 Fix the known eval bug (P0.4) — quick
`eval_convnext_sweep.py:116` unpacks 3 values from `batch_to_device`, which now returns **4**
(`images, labels, zones, sample_weight`). Update the unpack to 4 values so any post-hoc sweep runs.

### 3.2 Regenerate the canonical split over all 156 patients (P0.1)
The current split covers only 91 patients and leaves ~31% of labeled zones idle. Build a **stratified,
patient-level** split over **all 156** patients (stratify by patient positivity: an eye is positive if
any zone is binary-1 — see `run_patient_cv.py::patient_labels`). Freeze it as the **new canonical**
split file (e.g. `splits/canonical_split_v2_156.json`) and use it everywhere going forward. Building
blocks already exist in `zone_dataset.py` (`make_patient_split`, `load_or_create_split`).

### 3.3 Audit & fix fovea geometry (P0.3)
Count `fovea_fallback=True` sidecars among training images and **exclude or re-register** them — their
zone masks are attached to the wrong pixels and corrupt both labels and masked pooling. Record the
count in the results note. (`diagnose_unmatched.py` at repo root is a precedent for a one-off audit
script.)

### 3.4 Extend the CV harness to 5-fold (P0.2)
`run_patient_cv.py` is 3-fold today. Run it (and the Option A version) at **`--folds 5`**, stratified
by patient positivity, on the new 156-patient population.

### 3.5 Reprocess the unverified-geometry sidecars, then re-audit (follow-up to 3.3)

**Why this is a prerequisite, not a convenience.** The 3.3 audit (`fovea_audit.py`, results in
`PHASE0_RESULTS.md`) found that geometry quality is only *known* for 306 of 772 images:

| status | images | zone rows | |
|---|---:|---:|---|
| confirmed fallback (bad geometry) | 57 | 570 | excludable now via `--exclude-fovea-fallback` |
| ok (crosshair detected) | 249 | 2,490 | trustworthy |
| **missing_key (UNVERIFIED)** | **466** | **4,660** | **sidecars predate the `fovea_fallback` field** |

So **60% of images have unmeasured fovea geometry.** Where geometry *is* known, ~19% (57/306) are
bad — image-center fallback places all 10 zone masks on the wrong pixels, so those zone labels are
attached to the wrong region. That corrupts **both** training (wrong pixel→label associations) **and
evaluation** (the metric is scored against wrong-pixel ground truth). If ~19% carries into the 466
unknowns, ~140 images (~18% of the data) are geometrically corrupted — and Option A's masked pooling
reads those zone masks *directly*, so it depends on correct geometry even more than the crop baseline.
Bottom line: skipping this leaves an asterisk on every Option A number ("measured against data with
≥18% unverified geometry"), and part of the 0.52 ceiling could be a **measurement artifact, not a
model limit**. Resolve it before betting the architecture decision on it.

**Do this before freezing the 3.2 split**, so the new 156-patient canonical split is defined over
geometry-verified data.

**Step 1 — regenerate all sidecars with the flag (`--force`; idempotent, writes through the
`cleaned/` symlink so both pipelines update). A few minutes on 32 workers:**
```bash
cd /home/shashwat/per-zone-uveitis
PY=/home/shashwat/miniconda3/envs/venv/bin/python
$PY pre_processing.py \
  --data-dir data \
  --output-dir processed_image_arrays_multiclass \
  --annotations-xlsx "data/UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx" \
  --multiclass-zone-labels \
  --workers 32 \
  --force
```
*(Targeted alternative if you don't want to re-inpaint all 772: delete only the `.json` sidecars
listed as `missing_key` and run the same command **without** `--force` — the cache-miss path
reprocesses exactly those FPs. Brute `--force` is simpler and safe.)*

**Step 2 — re-audit and record the real number:**
```bash
$PY fovea_audit.py            # rewrites fovea_fallback_images.txt + prints the new breakdown
```
Update `PHASE0_RESULTS.md` with the post-reprocess counts (there should now be **zero** `missing_key`).

**Step 3 — decide exclusion.** The now-complete confirmed-fallback set is dropped everywhere by
`train_convnext.py --exclude-fovea-fallback` (and must be honored by the Option A trainer too). Note
the true clean-data image/patient count — this, not 772/156, is the real denominator for §4 and §6.
Re-registering (fixing) the bad-geometry images is a larger, separate task; excluding them is the
Phase 0 move.

---

## 4. Re-baseline the crop ConvNeXt under 5-fold CV (the honest denominator)

Before Option A, recompute the baseline on the **new split / 156 patients / 5-fold** so the comparison
is CV-to-CV. Use the existing trainer via the CV harness. The r14 config is the reference recipe
(ConvNeXt-Tiny, focal γ=3, inverse weights, exclude-tier1). Example (adjust once the harness is
5-fold and points at the new population):

```bash
cd /home/shashwat/per-zone-uveitis
export CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU
$PY run_patient_cv.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --folds 5 \
  --epochs 20 \
  --output-dir runs/INV01_baseline_crop_convnext_cv5
```

Result lands in `runs/INV01_baseline_crop_convnext_cv5/cv_summary.json` (mean ± std macro-F1 and
class-1 recall). **This mean is the number Option A must beat by more than the fold std.** Note the CV
harness runs folds with `--no-wandb` by design; the per-fold `metrics.json` and `cv_summary.json` are
the source of truth.

Reference single-model command (what each fold effectively runs) — useful for a one-off sanity run:
```bash
$PY train_convnext.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --split-json splits/canonical_split_v2_156.json \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 \
  --epochs 20 --batch-size 32 --num-workers 8 \
  --output-dir runs/INV01_baseline_singlesplit \
  --wandb-project uveitis-per-zone --wandb-run-name INV01_baseline_singlesplit
```

---

## 5. Build Option A (`train_zone_attention.py`)

Create a **new** training script — do not mutate `train_convnext.py` (keep the baseline runnable).
Implement the pieces from `DESCRIPTION.md §3.2`:

1. **New dataset** yielding `(full_image, zone_masks[10,H,W] at feature resolution, labels[10],
   zone_geometry[10,...])` per patient-eye. Reuse `extract_zones.make_zone_mask` for pixel-space
   wedges, then downsample to the pooling grid. **Mask out zones with no label for that eye.**
2. **Masked average pooling** `[B,C,H,W] × [B,10,H,W] → [B,10,C]`, with a guard for empty masks
   (normalize by mask sum + ε; never divide by 0 → NaN). See `DESCRIPTION.md §4`.
3. **One small attention layer** (1–2 heads) or ABMIL gated attention over the 10 tokens.
4. **Geometry-based positional encoding** into that attention (angle/radius/adjacency), **not** a bare
   learned index. (`DESCRIPTION.md §3.3 detail 2`.)
5. **Shared per-zone head** → `[B,10,2]`; **summed per-zone loss** using `losses.build_loss` with
   **focal γ=3 + inverse class weights** (kept from what works).
6. **Pool from a high-resolution stage (stride 8–16) or FPN, and/or run at ~512px** — the make-or-break
   knob (`DESCRIPTION.md §3.3 detail 1`). If inner zones map to 0 cells, the model can't see them.
7. **Wandb wiring** copied from `train_convnext.py`: `--wandb-project uveitis-per-zone`,
   `--wandb-run-name`, `--no-wandb`; log per-epoch train/val metrics, final test metrics, and — the
   payoff — **attention weights per zone** for a few eyes.

Keep the rest of the recipe identical to the baseline where possible (AdamW, backbone lr = head lr ×
0.1, cosine schedule, AMP, grad-clip 1.0, early stop on val macro-F1) so the comparison isolates the
architecture change.

**Verify before a full run:** do a 1–2 epoch smoke run on one fold with `--no-wandb`, print the pooled
token shapes and per-zone mask cell counts, and confirm **no zone has 0 active cells** and **no NaN
loss**. This is where detail (1) fails silently — catch it here.

---

## 6. Train Option A under 5-fold CV

Add an Option A path to the CV harness (or a parallel `run_patient_cv_zone_attention.py`) that launches
`train_zone_attention.py` per fold on the **new 156-patient split**, 5-fold, stratified by positivity.
Example shape:

```bash
cd /home/shashwat/per-zone-uveitis
export CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU
$PY run_patient_cv.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --folds 5 --epochs 20 \
  --trainer train_zone_attention.py \        # (add this arg when wiring in the new trainer)
  --output-dir runs/INV01_optionA_zone_attention_cv5
```

For an individual full-wandb run of one configuration (e.g. to inspect curves/attention on W&B):
```bash
$PY train_zone_attention.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --split-json splits/canonical_split_v2_156.json \
  --loss focal --focal-gamma 3.0 --class-weighting inverse --exclude-tier1 \
  --image-size 512 \                          # high-res for inner-zone support
  --epochs 20 --batch-size 16 --num-workers 8 \
  --output-dir runs/INV01_optionA_singlesplit \
  --wandb-project uveitis-per-zone --wandb-run-name INV01_optionA_singlesplit
```
(Batch size may need to drop at 512px; the A6000's 49 GB gives headroom.)

**Long runs:** launch in the background and tee the log, mirroring the existing scripts
(`research_loop.sh`, `run_cv3_r14.sh`):
```bash
mkdir -p runs/INV01_optionA_zone_attention_cv5
nohup $PY run_patient_cv.py ... > runs/INV01_optionA_zone_attention_cv5/cv.log 2>&1 &
```

---

## 7. Read the result

- **Primary comparison:** `runs/INV01_optionA_zone_attention_cv5/cv_summary.json` (Option A) vs
  `runs/INV01_baseline_crop_convnext_cv5/cv_summary.json` (re-baselined crop). Compare **mean macro-F1
  ± std** and **mean class-1 recall ± std**.
- **W&B:** per-epoch curves and attention visualizations for single-split runs live in the
  `uveitis-per-zone` project.
- **Per-fold artifacts:** each fold writes `runs/<...>/fold{k}/metrics.json`, `train.log`, `best.pt`,
  `split.json`.

---

## 8. Judge it honestly (success criterion)

Option A **wins only if** (from `DESCRIPTION.md §8`):

1. Its **5-fold CV macro-F1 beats the re-baselined crop CV macro-F1 by more than the fold std**, and
2. Its **class-1 recall is clearly above ~0.40** (the historical rut).

**Single-split gains do not count.** If Option A — built with high-resolution pooling and
geometry-aware positional encoding, on clean (fovea-audited) data over all 156 patients — does **not**
clear that bar, the conclusion is that the ceiling is **data/label quality, not architecture**, and
the next move is **labeling/data work, not more modeling**. Record that outcome as a real finding.

If it **does** clear the bar, the next experiment in this investigation is **Option D** (auxiliary
patient-level head; `DESCRIPTION.md §6`) — create a new experiment sub-folder for it.

---

## 9. Where to write results

Append a short results note with the CV numbers (mean ± std for both models), the fovea-fallback
count, the W&B run URLs, and the pass/fail on §8 — either at the bottom of `DESCRIPTION.md` or in a
`RESULTS.md` in this folder. Also update the project-level `RESEARCH_LOG.md` if the run belongs in the
protocol history. Keep this folder the single source of truth for the Option A investigation.
