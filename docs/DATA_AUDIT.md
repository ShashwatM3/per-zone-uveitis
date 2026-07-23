## Section 0 — Headline findings

- **7.38% of cleaned FPs (57/772) have `fovea_fallback: true`**, putting **570/7720 (7.38%) training-table rows** on geometrically wrong zone masks; **47/156 patients (30.13%)** have ≥1 fallback visit.
- Sidecar field `yellow_pixels` is **forced to 0 on every fallback** (`pre_processing.py` overwrite). The histogram bin `0 → 57` is that write artifact — re-running detection shows fallback FPs still have millions of yellow pixels; failures are **crosshair geometry** (44 cannot separate H/V lines; 13 fovea out of bounds), not missing yellow.
- On a 30-visit both-modality sample, **yellow overlay is on the FP** (FP yellow median 3,473,120 vs FA 7,998; 30/30 FP wins). Section B fallbacks are **not** explained by detecting on the wrong modality.
- FA is **not consumed by any training/eval code path** (only `MODALITY_PATTERN` skips non-FP in preprocessing). Every visit folder has ≥1 FA file (424/424), typically 1–2 files/eye with indices **0 (mode L)** and/or **1 (mode RGB)** — not a timestamped multi-phase sequence (no EXIF/phase metadata).
- Binary and multiclass tables share the **same 772 images / 7720 rows**; multiclass keeps tiers `{0:4770, 1:1171, 2:1779}` while binary merges `{1,2}→1` → `{0:4770, 1:2950}`.
- **84 images** (30 FP + 54 FA) are **3900×3072**, not 4000×4000 (patients 011–014, 016, 029, 039–040, 048, 053–055, 070).
- FP channel stats (30-image sample, content pixels): **R≈83, G≈65, B≈1.1** — consistent with Optos-style red/green pseudocolor (near-zero blue).
- Reported **0.519 ± 0.012** traces to local artifact `runs/protocol_cv3_r14_config/cv_summary.json` (exact mean 0.519285…, std 0.012084…); CV harness used `--no-wandb`. **No W&B run ID** found for that CV or for single-split r14 (`train.log` missing).

## Section A — Corpus census

- Raw `data/` absolute path: `/home/shashwat/per-zone-uveitis/data`
- `processed_image_arrays/` absolute path: `/home/shashwat/per-zone-uveitis/processed_image_arrays`
- Unique patients: **156**
- Total visits (patient date-folders): **424**
- Visits per patient: min **1**, median **2.0**, max **13**
- Patients with exactly 1 visit: **72**; with >1 visit: **84**
- Total FP images: **792**; total FA images: **1081**
- Visit–eye pairs with BOTH FP and FA: **790**
- Eye breakdown **before standardization** (filename OD/OS, all raw images): OD **927**, OS **945**
  - FP only: OD **393**, OS **399**
  - FA only: OD **534**, OS **546**
- Rows in `processed_image_arrays/zone_training_table.csv` (binary): **7720** (labels {'0': 4770, '1': 2950})
- `processed_image_arrays_multiclass/` **exists** at `/home/shashwat/per-zone-uveitis/processed_image_arrays_multiclass`
  - Rows: **7720** (same row count and same `Cleaned_Image` set as binary; verified equal)
  - Label distribution multiclass: {'0': 4770, '1': 1171, '2': 1779} ({'0': 61.7876, '1': 15.1684, '2': 23.044})
  - Label distribution binary: {'0': 4770, '1': 2950} ({'0': 61.7876, '1': 38.2124})
  - Difference: multiclass preserves raw tiers 0/1/2; binary maps `{1,2}→1` (0 unchanged). Tier1=1171, tier2=1779 sum to binary class1=2950.

## Section B — Zone geometry integrity

- Total sidecars scanned: **772** (under `processed_image_arrays/cleaned/`, shared via symlink with multiclass)
- `fovea_fallback == true`: **57** (7.3834%)
- `fovea_fallback == false`: **715**
- missing key / unreadable: **0** / **0**
- Patient-level: **47** / **156** patients (30.1282%) have ≥1 fallback sidecar
- Patients with ≥1 fallback visit (patient_id → fallback sidecar count):

| patient_id | fallback_count |
|---:|---:|
| 11 | 4 |
| 14 | 1 |
| 16 | 1 |
| 30 | 1 |
| 35 | 1 |
| 36 | 1 |
| 37 | 2 |
| 39 | 1 |
| 43 | 1 |
| 52 | 1 |
| 53 | 1 |
| 54 | 1 |
| 55 | 1 |
| 56 | 1 |
| 58 | 1 |
| 62 | 2 |
| 68 | 1 |
| 71 | 1 |
| 73 | 1 |
| 76 | 1 |
| 80 | 1 |
| 81 | 1 |
| 83 | 1 |
| 85 | 3 |
| 86 | 1 |
| 91 | 1 |
| 93 | 1 |
| 98 | 1 |
| 99 | 1 |
| 302 | 1 |
| 307 | 1 |
| 309 | 1 |
| 311 | 1 |
| 318 | 1 |
| 320 | 1 |
| 326 | 1 |
| 328 | 1 |
| 329 | 2 |
| 330 | 1 |
| 331 | 2 |
| 335 | 1 |
| 337 | 1 |
| 343 | 2 |
| 346 | 1 |
| 352 | 1 |
| 357 | 1 |
| 359 | 1 |

- Distribution of sidecar field `yellow_pixels`:
  - min **0**, max **9544416**, median **3513250.0** (n=772)
  - histogram bins:

| bin | count |
|---|---:|
| 0 | 57 |
| 1-199 | 0 |
| 200-1000 | 0 |
| 1001-5000 | 0 |
| 5001-20000 | 0 |
| >20000 | 715 |

- **Note (measured):** on fallback, `pre_processing.py` sets `yellow_count = 0` in the sidecar regardless of actual yellow. Re-detection on the 57 fallback FPs: **0** had FP yellow < 200; failure reasons were **cannot separate H/V crosshair lines (44)** and **fovea outside image bounds (13)**.
- Zone_Label cross-tab (multiclass table), fallback vs non-fallback visits:

| group | rows | tier0 | tier1 | tier2 | %0 | %1 | %2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fallback | 570 | 368 | 68 | 134 | 64.5614 | 11.9298 | 23.5088 |
| non-fallback | 7150 | 4402 | 1103 | 1645 | 61.5664 | 15.4266 | 23.007 |

- **Class skew:** fallback visits are **not strongly class-skewed**. Tier-0 is slightly higher (64.5614% vs 61.5664%), tier-1 slightly lower (11.9298% vs 15.4266%), tier-2 nearly identical (23.5088% vs 23.007%).
- Rows of `zone_training_table.csv` from fallback visits: **570** (same for binary and multiclass)
- `splits/canonical_split.json` fallback patients:
  - train: **19** / 63 patients — 30(1), 35(1), 36(1), 37(2), 43(1), 52(1), 54(1), 55(1), 58(1), 62(2), 68(1), 73(1), 81(1), 83(1), 85(3), 86(1), 91(1), 93(1), 98(1)
  - val: **3** / 14 — 16(1), 56(1), 71(1)
  - test: **6** / 14 — 11(4), 14(1), 39(1), 76(1), 80(1), 99(1)
  - 19 fallback patients are outside the 91-patient canonical split (53, 302, 307, 309, 311, 318, 320, 326, 328, 329, 330, 331, 335, 337, 343, 346, 352, 357, 359).

## Section C — Where does the yellow crosshair actually live?

Random sample: **30** visit–eye pairs with both modalities (seed=42). Yellow mask exactly as specified.
FA file preference: `*_FA_0000*` when present, else first sorted FA.

| visit | FP yellow | FA yellow |
|---|---:|---:|
| Patient329/20200922/OS | 2724854 | 19261 |
| Patient37/20240326/OS | 4583508 | 0 |
| Patient16/20250401/OS | 3818307 | 0 |
| Patient356/20240430/OS | 2953209 | 19294 |
| Patient62/20230926/OD | 2311645 | 0 |
| Patient58/20220301/OS | 3524585 | 0 |
| Patient56/20210309/OD | 2606350 | 0 |
| Patient40/20211221/OD | 3706811 | 0 |
| Patient353/20260203/OD | 4301900 | 18926 |
| Patient37/20210831/OD | 4794377 | 0 |
| Patient336/20220809/OD | 1370844 | 19168 |
| Patient355/20250603/OS | 2159021 | 18819 |
| Patient301/20260203/OD | 3582137 | 19310 |
| Patient35/20201124/OD | 3260085 | 0 |
| Patient317/20220531/OS | 2044454 | 19418 |
| Patient86/20210406/OD | 3120547 | 15995 |
| Patient20/20230411/OS | 2680782 | 0 |
| Patient19/20220816/OD | 3594300 | 0 |
| Patient35/20250624/OS | 2585359 | 0 |
| Patient56/20201110/OS | 3045596 | 0 |
| Patient56/20220823/OD | 2517706 | 0 |
| Patient94/20250311/OS | 4311182 | 19337 |
| Patient322/20221118/OS | 3582166 | 18678 |
| Patient17/20250414/OS | 4612880 | 0 |
| Patient306/20251028/OD | 3687678 | 19296 |
| Patient54/20200922/OD | 8137854 | 0 |
| Patient349/20260326/OS | 2727654 | 19348 |
| Patient330/20210316/OS | 3421654 | 19030 |
| Patient343/20241029/OS | 4251029 | 19129 |
| Patient85/20250805/OS | 3907913 | 18914 |

- FP wins: **30**; FA wins: **0**; ties: **0**
- Median FP yellow: **3473119.5**; median FA yellow: **7997.5**
- **Verdict: Overlay is on FP**
- FA yellow counts of 0 vs ~19k track FA index/mode: index **0** files are mode `L` (0 yellow after RGB convert); index **1** files are mode `RGB` with ~16k–20k yellow-ish pixels — orders of magnitude below FP overlay counts (millions), so not the annotation crosshair.
- **Implication for Section B:** because the overlay is on FP (the file preprocessing actually opens), the 7.38% fallback rate is **not** an artifact of searching the wrong modality. Fallbacks are crosshair-geometry failures on FPs that still contain abundant yellow.

## Section D — FA availability and phase structure

- Visits with ≥1 FA file: **424** / **424** (fraction **1.0**)
- FA files per visit–eye (among visit–eyes with ≥1 FA): min **1**, median **1.0**, max **3**; distribution {'2': 286, '1': 505, '3': 1}
- FA files per visit folder (both eyes): {2: 260, 4: 129, 1: 30, 3: 5} (measured)
- Observed FA filename indices matching `_FA_<n>`: range **0–1**; counts {'1': 789, '0': 286} (plus 6 files with alternate stem order `..._FA_OD_...` / `..._FA_OS_...` that still parse as FA)
- FP indices observed: range **0–1**; counts {'0': 787, '1': 1}
- **Single vs multi-frame:** per eye this is a **small companion set (typically 1–2 files), not a timed multi-phase FA sequence**.
  Evidence:
  - Indices only 0 and 1 (corpus-wide: index0=288 all mode `L`; index1=793 with 792 RGB + 1 L).
  - Sampled FA files have **empty EXIF** and empty PIL `info` (no acquisition phase/timestamp).
  - No sidecar/filename token encoding phase timing beyond the `_0000`/`_0001` index.
  - One visit–eye has 3 FA paths due to duplicate `FA_0001.tif` + `FA_0001.png` (Patient042/20200915/OD), not a third phase.
- **Code paths that read FA image files:** **none** found among repo `.py` files beyond `MODALITY_PATTERN` (used in `pre_processing.py` to **skip** non-FP when building the FP cache). `diagnose_unmatched.py` only references the annotations xlsx filename. No `Image.open` / array load of FA pixels in training, eval, or dataset code.

## Section E — Label distribution

Source: `processed_image_arrays_multiclass/zone_training_table.csv`.

- Overall tiers: 0=**4770** (61.7876%), 1=**1171** (15.1684%), 2=**1779** (23.044%); n=**7720**
- Per-zone 10×3 counts:

| zone | tier0 | tier1 | tier2 | %0 | %1 | %2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 482 | 146 | 144 | 62.4352 | 18.9119 | 18.6528 |
| 2 | 480 | 143 | 149 | 62.1762 | 18.5233 | 19.3005 |
| 3 | 490 | 140 | 142 | 63.4715 | 18.1347 | 18.3938 |
| 4 | 483 | 143 | 146 | 62.5648 | 18.5233 | 18.9119 |
| 5 | 491 | 97 | 184 | 63.601 | 12.5648 | 23.8342 |
| 6 | 485 | 111 | 176 | 62.8238 | 14.3782 | 22.7979 |
| 7 | 463 | 92 | 217 | 59.9741 | 11.9171 | 28.1088 |
| 8 | 458 | 103 | 211 | 59.3264 | 13.342 | 27.3316 |
| 9 | 464 | 113 | 195 | 60.1036 | 14.6373 | 25.2591 |
| 10 | 474 | 83 | 215 | 61.399 | 10.7513 | 27.8497 |

- Zones ≥95% a single class: **none** ([])
- Patients with zero positive zones (all labels 0 across all their rows): **38** / **156**
- Annotation spreadsheet columns (primary used by preprocessing):
  - File: `/home/shashwat/per-zone-uveitis/data/UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx`
  - Non-empty columns: `UWFFA`, `UWFFP`, `Patient_ID`, `Eye`, `Visit_Date`, `Image_File(FA)`, `Zone1_label`…`Zone10_label`, `Note for PE`
  - Secondary workbook `UWF_FP_Annotations_2.8.2026 Names removed.xlsx` columns: `UWFFA`, `UWFFP`, `Patient_ID`, `Eye`, `Visit_Date`, `Image_File(FA)`, `Zone1_label`…`Zone10_label` (no `Note for PE`)
- Multiple graders / confidence / uncertainty columns: **none present**. No column names matching grader/rater/confidence/uncertainty. `Note for PE` has 1 nonempty cell (`Is it FA or other Modality?`).

## Section F — FA↔FP registration check

20 random both-modality visits (seed=123). Method: grayscale → 500×500; content mask `gray>4`; FFT phase-correlation translation of FA mask onto FP mask; offsets scaled to FA native resolution; Pearson of FA intensity vs negated FP green on overlap after shift.

| visit | offset_orig_px (dy, dx) | IoU before | IoU after | Pearson FA vs −FP green |
|---|---|---:|---:|---:|
| Patient26/20220125/OS | (712, 80) | 0.5328 | 0.8759 | 0.253991 |
| Patient61/20220705/OD | (-1168, -96) | 0.3910 | 0.0753 | -0.148438 |
| Patient35/20201124/OD | (88, 40) | 0.5108 | 0.5201 | -0.652499 |
| Patient364/20230530/OS | (0, -8) | 0.6916 | 0.6913 | 0.034686 |
| Patient84/20240618/OD | (24, 64) | 0.6773 | 0.7117 | -0.210928 |
| Patient61/20220601/OD | (72, 64) | 0.5850 | 0.5880 | -0.315556 |
| Patient37/20230725/OD | (-400, 368) | 0.5502 | 0.7036 | -0.075744 |
| Patient24/20230117/OS | (-96, 128) | 0.6802 | 0.6745 | -0.092703 |
| Patient81/20251216/OS | (56, -40) | 0.9440 | 0.9812 | -0.419398 |
| Patient300/20250603/OD | (56, 48) | 0.9271 | 0.9803 | -0.149495 |
| Patient306/20251028/OS | (136, -32) | 0.8894 | 0.9601 | -0.354358 |
| Patient72/20240221/OD | (48, 16) | 0.9445 | 0.9861 | -0.009022 |
| Patient76/20240409/OS | (64, -80) | 0.9066 | 0.9634 | 0.076631 |
| Patient43/20220830/OD | (-216, 80) | 0.7045 | 0.6913 | -0.4409 |
| Patient40/20210629/OS | (24, 8) | 0.7795 | 0.7863 | -0.620549 |
| Patient74/20240528/OD | (56, 48) | 0.9262 | 0.9795 | -0.564324 |
| Patient306/20251028/OD | (136, -40) | 0.8867 | 0.9621 | -0.38099 |
| Patient72/20240305/OS | (56, -48) | 0.8509 | 0.8686 | 0.089468 |
| Patient343/20241029/OS | (72, 24) | 0.9215 | 0.9783 | 0.06873 |
| Patient58/20220301/OD | (-24, 72) | 0.6028 | 0.6019 | -0.588072 |

- Median IoU before correction: **0.741974**
- Median IoU after correction: **0.827446**
- Median |offset| (px): **86.11**; max **1171.94**
- Median Pearson: **-0.180211**
- **Rigid translation sufficiency:** translation helps on average (median IoU 0.742 → 0.827), but is **not universally sufficient**. 2/20 pairs got worse after the estimated shift; 4/20 had |offset| > 200 px (one catastrophic: Patient61/20220705/OD offset (−1168, −96), IoU 0.391 → 0.075). Median Pearson is **negative** (−0.18), so intensity correspondence after translation is weak — consistent with residual rotation/scale/modality contrast differences beyond pure translation.

## Section G — Image properties

- Dimensions:
  - FP: {'(4000, 4000)': 762, '(3900, 3072)': 30} (n=792)
  - FA: {'(4000, 4000)': 1027, '(3900, 3072)': 54} (n=1081)
  - Deviations from 4000×4000: **84** files, all size **3900×3072**, patients: Patient011, Patient012, Patient013, Patient014, Patient016, Patient029, Patient039, Patient040, Patient048, Patient053, Patient054, Patient055, Patient070
- FP modes/formats: modes {'RGB': 792}; formats {'PNG': 790, 'JPEG': 2}
- FA modes/formats: modes {'L': 289, 'RGB': 792}; formats {'PNG': 1079, 'TIFF': 1, 'JPEG': 1}
- Bit depth: loaded arrays are **`uint8`** (8-bit) for sampled FP RGB, FA L, and FA RGB. PIL `Image.bits` is often `None` for PNG; inferred from mode/`numpy.dtype` after load.
- Channel count: FP all 3-channel RGB; FA mixed 1-channel L (289) and 3-channel RGB (792)
- Non-black fraction (`mean(RGB)>4` on ≤256px thumbnail):
  - FP: min **0.321520**, median **0.359142**, max **0.969010**
  - FA: min **0.129816**, median **0.315784**, max **0.967005**
- Optos-style pseudocolor check (30 random FP, content pixels only):

| channel | mean_of_means | std_of_means | mean_of_stds |
|---|---:|---:|---:|
| R | 83.3482 | 17.2737 | 40.6820 |
| G | 64.5802 | 14.2051 | 33.1751 |
| B | 1.1099 | 0.1956 | 1.6992 |

- Blue near zero with substantial R and G is consistent with Optos red/green laser pseudocolor.

## Section H — Baseline reproducibility

- Run r14 config located at `runs/protocol_r14_convnext_exclude_tier1/metrics.json` (and `RESEARCH_LOG.md`).
- Exact command (from RESEARCH_LOG / metrics args):

```bash
/home/shashwat/miniconda3/envs/venv/bin/python train_convnext.py \
  --csv processed_image_arrays_multiclass/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --epochs 20 --batch-size 32 --num-workers 8 \
  --lr 0.0001 --weight-decay 0.0001 --image-size 288 --seed 13 \
  --zone-embed-dim 64 --head-hidden 256 --patience 8 --grad-clip-norm 1.0 \
  --split-json splits/canonical_split.json \
  --output-dir runs/protocol_r14_convnext_exclude_tier1 \
  --wandb-project uveitis-per-zone
```

- Hyperparameters from `metrics.json` args: loss=focal, focal_gamma=3.0, class_weighting=inverse, exclude_tier1=true, epochs=20, batch_size=32, lr=1e-4, weight_decay=1e-4, image_size=288, seed=13, zone_embed_dim=64, head_hidden=256, patience=8, grad_clip_norm=1.0, num_workers=8, positive_oversample_factor=1.0, soft_labels=false, no_pretrained=false, no_amp=false. Best epoch **10**; test macro-F1 **0.55992078**.
- `run_patient_cv.py` implements:
  - **K-fold patient-level** CV via `StratifiedKFold` on patients (default **3** folds, `--folds`)
  - Patients grouped by `patient_id`; stratification label = 1 if any zone binary-positive else 0
  - Seed: `--seed` default **13** for `StratifiedKFold(random_state=seed)`; within each fold, train/val split uses `seed + fold` with 15% val of the train_val patients
  - Each fold invokes `train_convnext.py` with r14 recipe (focal γ=3, inverse weights, `--exclude-tier1`) and **`--no-wandb`**
- **0.519 ± 0.012 traceability:** YES — local artifact `/home/shashwat/per-zone-uveitis/runs/protocol_cv3_r14_config/cv_summary.json` with `test_macro_f1_mean=0.5192852385041827`, `test_macro_f1_std=0.012083624817819785`, per-fold F1 `[0.532324678021797, 0.5223311209693134, 0.5031999165214377]`. Also fold dirs `fold0/`, `fold1/`, `fold2/` with `metrics.json`. **No W&B run IDs** (harness passes `--no-wandb`). Single-split r14: local `runs/protocol_r14_convnext_exclude_tier1/{metrics.json,best.pt}`; `train.log` absent; **W&B run ID UNKNOWN** (not in `WANDB_RUNS.md`; local `wandb/run-*` pruned).
- W&B project names referenced in repo: **`uveitis-per-zone`** only (defaults in `train_convnext.py`, `train_clip_convnext.py`, `train_retfound.py`; scripts/docs).

## Section I — Environment

- GPUs (nvidia-smi / torch):
  - 0: NVIDIA RTX A6000, 49140 MiB (~47.54 GiB)
  - 1: NVIDIA GeForce RTX 3090, 24576 MiB (~23.69 GiB)
  - 2: NVIDIA GeForce RTX 3090, 24576 MiB (~23.69 GiB)
  - 3: NVIDIA RTX A6000, 49140 MiB (~47.53 GiB)
- CUDA: toolkit `nvcc` **11.5** (V11.5.119); PyTorch built with CUDA **12.8** (`torch 2.10.0+cu128`)
- Free disk on data volume (`df -h` on `/home/shashwat/per-zone-uveitis/data`, filesystem `/`): **400G available** (1.8T total, 1.4T used, 78%)
- Conda env name: **`venv`** at `/home/shashwat/miniconda3/envs/venv/bin/python`
- `torch.cuda.is_available()`: **True**
- Git: branch **`feature/clip-convnext`**, HEAD **`80b4297ee8d75b487709d5abacba3a2d01bf5884`**
- Local branches: `feature/binary-zone-classification`, `feature/clip-convnext`, `main`
- Remote branches: `origin/HEAD -> origin/main`, `origin/feature/binary-zone-classification`, `origin/feature/clip-convnext`, `origin/feature/patient-aggregation`, `origin/main`
- `timm` installed: **yes**, version **1.0.25**

## Open questions

- **W&B run ID for single-split r14:** UNKNOWN — `train.log` missing and local `wandb/run-*` pruned; cloud search on project `uveitis-per-zone` by run name/date would be needed.
- **W&B run IDs for the 3-fold CV that produced 0.519:** not applicable / UNKNOWN as W&B — harness used `--no-wandb`; only local `cv_summary.json` / fold `metrics.json` exist.
- **Semantic meaning of FA `_0000` (L) vs `_0001` (RGB):** UNKNOWN — no EXIF/phase metadata; would need acquisition protocol docs or clinician confirmation (grayscale angiogram vs colorized companion).
- **True yellow pixel counts at preprocessing time for fallbacks:** UNKNOWN from sidecars alone (field overwritten to 0); current on-disk FPs were remeasured in this audit, but historical file identity at cache-write time was not independently versioned.
- **Whether any of the 20 unmatched FPs (792−772) would change census if later matched:** listed in `processed_image_arrays_multiclass/unmatched_fps.txt`; not re-audited here beyond summary.json counts.
- **PIL `bits` metadata for PNG:** often `None`; bit depth inferred from array dtype. A PNG chunk-level bit-depth parse was not performed.

