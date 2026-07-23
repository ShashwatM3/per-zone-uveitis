# PHASE 2 — Downstream zone classification

**Status:** blocked by Gate 1.
**Gate:** `docs/ACCEPTANCE_GATES.md` §Gate 2 (sub-gates A, B)
**Ledger:** `KANBAN/PHASE_2_downstream/`

---

## 1. Objective

Take the Phase 1 pretrained encoder and predict the FA-derived severity grade for
each of the 10 zones from a **single FP image**. FA is not loaded at any point in
this phase.

---

## 2. This is the global pipeline, not the per-zone crop pipeline

Explicit, because the two coexist in this repo's history and must not be mixed.

| | Legacy per-zone crop (r01–r26) | **Phase 2 (this spec)** |
|---|---|---|
| Forward passes | 10 per image, one per zone crop | **1 per image** |
| Zone isolation | Black-masked pie-slice crop, resized | **Masked pooling over the feature map** |
| Cross-zone context | Deleted | **Preserved** |
| Zones treated as | 10 i.i.d. samples | **10 outputs of one acquisition** |
| Augmentation | Independent per crop (incorrect) | **Shared across zones automatically** |

The context argument is not speculative: the `patient-aggregation` branch tried
to restore context *post hoc* by training an aggregator on frozen per-zone
features and was ceiling-capped. Context deleted at the input cannot be recovered
downstream.

Phase 1's pretext is also whole-image. Switching to crops here would introduce a
pretrain/fine-tune scale mismatch on top of everything else.

---

## 3. Architecture

```
FP ──► crop to content bbox ──► resize 1120 ──► E_θ (+m_FP) ──► 70×70×C features
                                                                      │
                                    zone masks (Phase 0 geometry) ────┤
                                                                      ▼
                                              per-zone: masked mean ⊕ masked max
                                                                      │
                                                      ⊕ zone-identity embedding
                                                                      ▼
                                                        shared MLP head ──► K logits
```

**Encoder.** Phase 1's `E_θ` with the FP modality embedding. The FA embedding and
the predictor are discarded.

**Zone masks.** Call `extract_zones.make_zone_mask(W, H, cx, cy, zone, angle_deg)`
with the **Phase 0 corrected geometry**, then downsample to the 70×70 feature grid.
Do not reimplement the geometry. Do not crop.

**Pooling — masked mean *concatenated with* masked max.** Not mean alone.

Rationale: leakage in these FAs is punctate and multifocal, not a uniform wash.
Over a peripheral quadrant of ~350 positions, a low-severity zone with ~10
leakage-bearing positions shifts the mean by ~3% — nearly indistinguishable from a
clean zone, and easily swamped by inter-patient illumination and pigmentation
variance. A high-severity zone with ~50 shifts ~14% and separates cleanly. **Mean
pooling measures areal extent and nothing else.** Max recovers peak focal
intensity. Keep both; head input becomes 2C.

**Zone-identity embedding.** A learned per-zone vector (dim 64, matching the
legacy `--zone-embed-dim`) concatenated to each pooled vector. This lets a shared
head specialise per zone without paying for 10 independent heads, and relaxes the
assumption that the severity signal is identical across zones.

**Head.** Shared MLP `2C + 64 → 256 → K`, matching the legacy `--head-hidden 256`.

---

## 4. Label formulation

Corpus label distribution (multiclass table): tier 0 = 4770 (61.8%),
tier 1 = 1171 (15.2%), tier 2 = 1779 (23.0%).

**Run binary first, for gate comparability.** Gate 2.A compares against the
Phase 0 baseline, which is the legacy r14 recipe: binary labels with
`--exclude-tier1`. The first Phase 2 run must match that label definition exactly,
or the comparison is invalid.

**Then run 3-class ordinal as a second investigation.** Use CORN ordinal loss:
K=3 decomposes into two rank-consistent binary tasks, `P(y>0)` and `P(y>1)`
trained conditionally on `y>0`. Grades 0/1/2 are ordered; nominal cross-entropy
treats them as unrelated categories, forcing the middle grade to be discovered as
an isolated island in logit space.

**Important caveat on prior evidence.** The legacy finding that `--exclude-tier1`
*helped* was measured on ~99%-corrupted geometry. It is not reliable evidence that
tier 1 is inherently ambiguous. Re-establish that question on corrected data
before treating tier-1 exclusion as settled.

Class weighting: inverse-frequency, as in the legacy recipe. Focal loss remains
available as a flag for ablation but is not the default — with corrected geometry
the class-imbalance problem may look materially different.

---

## 5. Fine-tuning schedule

```
epochs 0–10 : encoder frozen, head only, lr 1e-3
epochs 10+  : last encoder stage unfrozen @ lr 1e-5; head @ 1e-4
early stopping on validation macro-F1, patience 10, epoch ceiling 100
```

Freeze-then-unfreeze rather than end-to-end from step 0: with 156 patients, an
immediately-unfrozen encoder will overfit before the randomly-initialised head has
produced a useful gradient signal.

**Assert and log the trainable-parameter count at both schedule transitions.**
The known `requires_grad` regression cut trainable parameters from ~15.6M to ~98K
silently. Fail the run on a >20% deviation from expected.

---

## 6. Augmentation

Per `docs/AUGMENTATION_POLICY.md`. In summary for this phase:

- **No geometric augmentation.** Rotation would invalidate the zone masks, and
  `make_zone_mask`'s ONH placement bug makes it unsafe even with transformed
  geometry until that fix is verified.
- Mild photometric jitter (brightness, contrast). No hue shifts — Optos channel
  identity encodes imaging depth.
- CLAHE deterministic, identical parameters to Phase 1.
- No MixUp, no CutMix. Both were tried in the legacy sweep and flatlined, and both
  destroy the spatial-label correspondence this architecture depends on.
- The existing OD/OS flip is a fixed standardization step. Do not add a random
  horizontal flip on top of it.

---

## 7. Evaluation

**Harness: `run_patient_cv.py` and `zone_dataset.py::patient_split`, used
verbatim.** Not a reimplementation, not a variant. A comparison run on a different
harness is not a comparison. If the new model needs a different dataloader, wrap
it — do not fork the split logic.

- 3 folds, patient-grouped, ≥3 seeds
- W&B enabled with `--require-wandb` (the legacy CV harness ran `--no-wandb`;
  this must be fixed)

Report, per fold and as mean ± std:
- macro-F1 (primary)
- **per-class F1 and recall, every class listed explicitly**
- quadratic-weighted Cohen's κ (for the 3-class ordinal arm)
- per-zone macro-F1 — does peripheral performance differ from posterior?
- AUROC, per-class specificity, ROC curves, confusion matrix (the existing
  `train_convnext.py` W&B logging already covers most of this and should be reused)

Additionally, and specific to this project: report macro-F1 **stratified by Phase 0
geometry confidence** (`od_containment_pass`, transfer residual). If performance
is markedly better on high-confidence geometry, that quantifies the residual
registration bound and tells you where the next gain is.

---

## 8. Passing

Gate 2.A: CV macro-F1 mean exceeds the Phase 0 clean baseline by ≥0.02 with
non-overlapping ±1 std.

Gate 2.B: the gain is not carried entirely by the majority class. Positive-class
recall must not regress against the Phase 0 baseline.

A headline macro-F1 improvement with flat or falling positive-class recall is not
progress toward the stated clinical motivation, and does not pass.

---

## 9. On failure

Consult `docs/ACCEPTANCE_GATES.md` §Escalation and name the rung in
`NEXT_STEPS.md`.

The most likely diagnosis if Gate 1 passed but Gate 2 fails is **rung 2** — the
representation is sound and the problem is in pooling, head, or label definition.
Iterate here without retraining the encoder. Retraining Phase 1 in response to a
Phase 2 failure is the expensive mistake, and the frozen-probe result from Gate
1.B is what tells you it is unnecessary.
