# Research loop log (feature/clip-convnext)

Baseline best before loop: `clip_convnext_10ep_baseline` test.macro_f1=0.5251

Canonical split: `splits/canonical_split.json` (from baseline run, seed=13).

**Current best:** `protocol_r03_convnext_focal` test.macro_f1=**0.5285**

| Run | Name | Hypothesis | test.macro_f1 | Δ vs best | Decision |
|-----|------|------------|---------------|-----------|----------|
| — | (prior best) | — | 0.5251 | — | — |
| 1 | protocol_r01_convnext_soft_ce | Soft CE + multiclass soft labels | 0.4849 | -0.0402 | Below baseline → tried CLIP |
| 2 | protocol_r02_clip_default | CLIP Large-D defaults, binary 20ep | 0.5259 | +0.0008 | Slight gain → exp 3 compare |
| 3 | protocol_r03_convnext_focal | ConvNeXt-Tiny focal vs CLIP (exp 3) | **0.5285** | +0.0034 | **NEW BEST** — ConvNeXt wins on same split |
| 4 | protocol_r04_clip_focal | CLIP + focal loss | 0.4990 | -0.0295 | No gain (val OK, test weak) |
| 5–10 | (in progress) | CLIP tune + ConvNeXt zone128 + CLIP soft labels | — | — | resuming run 5… |
| 5 | (running) protocol_r05_clip_freeze3 | CLIP unfreeze backbone after 3 epochs | — | — | — |
| 5 | protocol_r05_clip_freeze3 | CLIP unfreeze backbone after 3 epochs | 0.5044 | -0.0240 | no gain |
| 6 | (running) protocol_r06_clip_lr2e4 | CLIP head LR 2e-4 | — | — | — |
| 11 | protocol_r11_convnext_focal_g3 | Branch B1: increase focal gamma to 3.0 on ConvNeXt | 0.5461 | +0.0176 | **NEW BEST** |
| 12 | protocol_r12_convnext_softlabels | Branch B3: soft labels + soft CE (multiclass CSV) | 0.4849 | -0.0612 | Regressed; keep r11 as best |

## Iteration 11 — 2026-05-28
**Config:** `train_convnext.py --loss focal --focal-gamma 3.0 --epochs 20 --split-json splits/canonical_split.json --output-dir runs/protocol_r11_convnext_focal_g3`
**Hypothesis:** Increasing focal gamma should emphasize hard minority examples and improve class-1 discrimination under class suppression.
**Result:** val macro-F1=0.5379, test macro-F1=0.5461, best_epoch=9
**Per-class:** class0 F1=0.6088 recall=0.5728, class1 F1=0.4834 recall=0.5271
**Diagnosis:** Branch B remains active because one per-class F1 (class1) is still <0.60 despite improved macro-F1.
**Next:** Run `--soft-labels --loss soft_ce` with multiclass CSV (Branch B3) to reduce ambiguity-driven label noise and recover class-1 F1.
**Literature cited:** N/A

## Iteration 12 — 2026-05-28
**Config:** `train_convnext.py --csv processed_image_arrays_multiclass/zone_training_table.csv --data-root processed_image_arrays_multiclass --soft-labels --loss soft_ce --epochs 20 --split-json splits/canonical_split.json --output-dir runs/protocol_r12_convnext_softlabels`
**Hypothesis:** Treating ambiguous zones as soft labels should reduce noise and improve suppressed class-1 performance.
**Result:** val macro-F1=0.5166, test macro-F1=0.4849, best_epoch=1
**Per-class:** class0 F1=0.6541 recall=0.7185, class1 F1=0.3156 recall=0.2681
**Diagnosis:** Branch B3 failed; class-1 collapsed further and overall macro-F1 regressed substantially.
**Next:** Re-anchor on `protocol_r11_convnext_focal_g3` as current best and continue diagnosis from STEP 1 before launching the next run.
**Literature cited:** N/A

## Iteration 13 — 2026-05-28
**Config:** `train_convnext.py --loss focal --focal-gamma 3.0 --positive-oversample-factor 2.0 --epochs 20 --split-json splits/canonical_split.json --output-dir runs/protocol_r13_convnext_focal_g3_os2`
**Hypothesis:** 2x positive oversampling might lift class-1 recall/F1 without sacrificing macro-F1.
**Result:** run manually terminated at epoch 9 due collapse pattern; no final test metrics recorded.
**Per-class:** by epoch 9 val class0 recall=0.1383, class1 recall=0.9379 (strong positive-class overprediction).
**Diagnosis:** Oversampling + focal gamma 3.0 + inverse class weighting over-biased toward class 1; run terminated early.
**Next:** Keep 14-epoch max going forward; test `focal_gamma=3.0` with `class_weighting=none` (remove one biasing mechanism) and no oversampling.
**Literature cited:** Small medical-image TL literature supports cautious imbalance handling and avoiding stacked reweighting that destabilizes minority/majority trade-offs.

### D1 Literature check (concise)
- Transfer-learning studies on small medical datasets consistently recommend simple adaptation first, then gradual complexity; over-aggressive reweighting often harms calibration.
- Uveitis/retinal works show strong AUC can coexist with poor class-wise operating points; threshold tuning helps only when probabilities are separable.
- In our checkpoint, threshold sweep on `protocol_r11_convnext_focal_g3` selected `0.50` as optimal (no improvement), so training dynamics—not thresholding—remain the bottleneck.
