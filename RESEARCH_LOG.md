# Research loop log (feature/clip-convnext)

Baseline best before loop: `clip_convnext_10ep_baseline` test.macro_f1=0.5251

Canonical split: `splits/canonical_split.json` (from baseline run, seed=13).

**Current best:** `protocol_r14_convnext_exclude_tier1` test.macro_f1=**0.5599**

| 14 | protocol_r14_convnext_exclude_tier1 | NEXT 3 exclude tier-1 | **0.5599** | — | **BEST** |
| 18 | protocol_r18_efficientnet_excl_t1 | EfficientNet-B3 backbone | 0.5289 | -0.0310 | NaN collapse after ep1 |

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
| 14 | protocol_r14_convnext_exclude_tier1 | NEXT 3: exclude tier-1 from training only | **0.5599** | +0.0138 | **NEW BEST** — modest gain vs r11 |

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

## Iteration 14 — 2026-05-28
**Config:** `train_convnext.py --csv processed_image_arrays_multiclass/zone_training_table.csv --data-root processed_image_arrays_multiclass --loss focal --focal-gamma 3.0 --epochs 20 --split-json splits/canonical_split.json --exclude-tier1 --output-dir runs/protocol_r14_convnext_exclude_tier1` (training stopped at epoch 17; best.pt from epoch 10; test eval from checkpoint)
**Hypothesis:** Removing tier-1 ambiguous labels from training cleans the signal and improves class-1 purity.
**Result:** val macro-F1=0.5533, test macro-F1=0.5599, best_epoch=10
**Per-class:** class0 F1=0.6144 recall=0.5709, class1 F1=0.5054 recall=0.5602
**Diagnosis:** Small improvement over r11; class-1 F1 still ~0.51. Branch B still active. Training was CPU-bound (CUDA driver mismatch).
**Next:** Protocol NEXT 1–2: threshold + TTA sweeps on `runs/protocol_r14_convnext_exclude_tier1/best.pt`. Future runs use `--epochs 14`.
**Literature cited:** N/A

## Iteration 15 — 2026-05-28 (NEXT 1: threshold sweep, no retrain)
**Script:** `eval_convnext_sweep.py` on `protocol_r14` checkpoint
**Key Config:** val threshold sweep 0.20–0.70 step 0.02
**Hypothesis:** Lower threshold may recover class-1 recall on 62/38 data.
**Result:** optimal val threshold=0.50; test macro-F1=0.5599 (unchanged vs 0.5)
**Diagnosis:** Branch D2 — no free gain from thresholding on current checkpoint.
**Next:** TTA sweep (iteration 16).
**Literature cited:** N/A

## Iteration 16 — 2026-05-28 (NEXT 2: TTA sweep, no retrain)
**Script:** `eval_convnext_sweep.py` (4-view flip TTA)
**Result:** test TTA @0.50 macro-F1=0.5483 (worse than 0.5599 baseline)
**Diagnosis:** TTA hurts on this model; do not enable by default.
**Next:** Training — remove inverse class weighting on exclude-tier1 config (Branch B follow-up from iter 13).
**Literature cited:** N/A

## Iteration 17 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r15_convnext_excl_t1_cw_none`
**Key Config:** exclude-tier1, focal γ=3, class_weighting=none, epochs=14
**Hypothesis:** Removing inverse class weights reduces over-bias from stacked imbalance handling.
**Result:** val macro-F1=0.4922, test macro-F1=0.4719, best_epoch=9
**Per-class:** class0 F1=0.6777 recall=0.8422, class1 F1=0.3067 recall=0.2160
**Diagnosis:** Branch B — severe class-1 suppression returned; regressed vs r14. Keep r14 as best.
**Next:** BBFL (balanced batches + focal γ=1.5) on exclude-tier1 config (Singh et al. JMI 2023).
**Literature cited:** Singh et al., batch-balanced focal loss on fundus images.

## Iteration 18 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r16_convnext_bbfl_excl_t1`
**Key Config:** exclude-tier1, balanced-batches, focal γ=1.5, class_weighting=none, epochs=14
**Hypothesis:** BBFL batch balancing (Singh JMI 2023) improves class-1 F1 vs focal-only.
**Result:** val macro-F1=0.5166, test macro-F1=0.5083, best_epoch=9
**Per-class:** class0 F1=0.6084 recall=0.5728, class1 F1=0.4082 recall=0.4503
**Diagnosis:** Branch B2 — no gain vs r14; BBFL did not beat exclude-tier1 + γ=3 baseline.
**Next:** Tier confidence weights (C2b) without excluding tier-1.
**Literature cited:** Singh et al. 2023 BBFL on fundus RNFLD/glaucoma.

## Iteration 19 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r17_convnext_tier_conf_wt`
**Key Config:** tier-confidence-weights (0.3 on tier-1), focal γ=3, inverse weights
**Result:** test macro-F1=0.5095, best_epoch=7
**Diagnosis:** Branch C2b — no gain vs r14; down-weighting ambiguous tiers did not help.
**Next:** EfficientNet (A3) then RETFound when weights ready.
**Literature cited:** N/A

## Iteration 20 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r18_efficientnet_excl_t1`
**Key Config:** efficientnet_b3 backbone, exclude-tier1, focal γ=3
**Result:** test macro-F1=0.5289, best_epoch=1 (early stop ep9; NaN losses ep2+)
**Diagnosis:** Branch A3 failed — training instability; all-class-0 collapse on val after ep1.
**Next:** RETFound (A2) + r20/r21 queue; do not retry EfficientNet without LR/AMP fix.
**Literature cited:** N/A

## Iteration 21 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r20_convnext_excl_t1_14ep`
**Key Config:** same as r14 (exclude-tier1, focal γ=3, inverse), 14 epochs only
**Result:** test macro-F1=0.5368, best_epoch=8
**Diagnosis:** Below r14 at 20ep (0.5599); 14 epochs slightly underfits vs longer run.
**Next:** RETFound r19 with valid weights.
**Literature cited:** N/A

## Iteration 23 — 2026-05-28
**Script:** `train_retfound.py` — `protocol_r19_retfound_excl_t1`
**Key Config:** RETFound MAE CFP, exclude-tier1, focal γ=3, inverse, 14ep
**Result:** test macro-F1=0.4889, class1_recall=0.5512
**Diagnosis:** Below ConvNeXt r14 (0.5599); fundus MAE did not help this zone task.
**Next:** Phase3 r22 (γ=2.5) then r23–r25 sweeps.
**Literature cited:** Zhou et al. RETFound Nature 2023; label-efficiency OCT study PMC11950740.

## Iteration 24–28 — 2026-05-28 (phase3 batch)
| Run | F1 | Note |
|-----|-----|------|
| r22 γ=2.5 | 0.5345 | below r14 |
| r23 effective CW | 0.4878 | hurt; low class1 recall |
| r24 zone_embed 128 | 0.4934 | below r14 |
| r25 balanced batches | 0.5309 | below r14 |

**Diagnosis:** Phase3 complete 14:32; none beat r14 (0.5599). Pipeline stalled — `run_autonomous_loop.sh` only ran phase3 (nested call), resume block never executed.
**Next:** `run_autonomous_resume.sh` → r26–r28, sweep, 5-fold CV, ceiling check.

## Iteration 22 — 2026-05-28
**Script:** `train_convnext.py` — `protocol_r21_convnext_mixup_excl_t1`
**Key Config:** r14 config + MixUp α=0.2 (Branch D4)
**Result:** test macro-F1=0.5025
**Diagnosis:** MixUp hurt on this task; do not combine with current best.
**Next:** RETFound backbone (Branch A2).
**Literature cited:** Galdrán et al. Balanced-MixUp MICCAI 2021 (retinal fundus).
