# ACCEPTANCE GATES & ESCALATION

Fixed criteria for "what counts as success" at each phase, and what class of
change to make when a gate fails. Every `NEXT_STEPS.md` must cite the gate it is
targeting and, on failure, the escalation rung it is invoking.

Nothing in this file may be renegotiated to make a run pass. If a gate is wrong,
change it in a dated edit **before** the run, never after.

---

## 0. Standing rules

- **The unit of evidence is a patient-grouped CV mean ± std across ≥3 folds.**
  Single-split numbers are inadmissible for any gate. Precedent: legacy r26 hit
  ~0.58 validation and ~0.45 test on the same config.
- **Minimum 3 seeds** for any run whose result will close an investigation.
- **Every gate metric must be traceable to a W&B run ID.**
- The legacy figure **0.519 ± 0.012 is not a baseline.** It was produced on
  ~99%-misplaced zone labels and 63%-inpainted images. It is a noise floor.
  The real baseline is established by Gate 0.C below.

---

## Gate 0 — Phase 0, geometry rebuild

Phase 0 is complete only when **all four** sub-gates pass.

### 0.A — Geometry source correctness
- FA crosshair detection (TIGHT mask) succeeds on **≥95%** of `_FA_0001` visit-eyes.
  *Audited actual: 768/789 = 97.3%. This gate confirms no regression.*
- Visit-eyes with no usable FA geometry are **excluded from the training table**,
  not silently defaulted. There must be no surviving centre-fallback path.

### 0.B — Anatomical validation (the hard gate)
The optic-disc containment test is the primary objective check.

- Detect the optic disc centroid on each FP independently (brightest coherent
  structure; see Phase 0 §4).
- Project zone 9 (ONH ellipse, radii `rx=80, ry=95`) using the transferred geometry.
- **Run the test under both ONH offset conventions** — unrotated
  (`cx + onh_offset_x`) and rotated by `angle_deg`. Report both containment rates.
  The higher one identifies the annotation tool's convention; document it and
  adopt it. See Phase 0 §6.
- **≥85% of images must have their detected OD centroid inside the zone-9 ellipse**
  under the winning convention.
- Report containment **stratified by zone group** (inner / ring / ONH / periphery),
  not as a single averaged number. An 85% overall pass can conceal
  "outer zones perfect, inner zones broken" — and the inner zones are exactly where
  an ~86 px transfer error is large relative to a 159 px radius.
- **Visual QC: ≥23/25 randomly sampled overlays judged anatomically correct by a
  human.** *Current pipeline scores 0/25. This is non-negotiable and cannot be
  automated away.*

### 0.C — Clean baseline established
- Regenerate the training table with corrected geometry.
- Re-run the exact r14 config (focal γ=3, inverse class weighting,
  `--exclude-tier1`, binary labels, image-size 288, seed 13) under
  `run_patient_cv.py` with 3 folds, **now with W&B enabled**.
- Record `test_macro_f1_mean ± std` and per-class recall.
- **This number replaces 0.519 as the project baseline.** There is no threshold
  to pass here — the gate is that the number exists, is logged to W&B, and is
  recorded in `KANBAN/PHASE_0_geometry/`.

### 0.D — Image integrity
- FP images are **not** inpainted (they carry no overlay; see Phase 0 §3).
- Any FP where a TIGHT-mask crosshair with balanced H/V counts *is* found must be
  flagged for manual review, not auto-inpainted.
- Median PSNR between raw FP and the cached array used for training must exceed
  **40 dB** (i.e. effectively lossless). *Current pipeline: 20.0 dB.*

---

## Gate 1 — Phase 1, cross-modal SSL pretext

### 1.A — Health checks (must pass before Phase 2 is attempted)
- **Feature-map coherence:** PCA of encoder spatial features, top-3 components to
  RGB, logged as W&B Images at ≥4 checkpoints. Structure must be spatially
  coherent (vessels / disc / macula separable), not speckle. Human judgement.
- **No collapse:** per-dimension standard deviation of the student's pooled
  embedding, logged every epoch, must stay above **0.05** for **≥95%** of
  dimensions. A collapsing run must abort, not finish.

### 1.B — Representation gate (the decision gate)
- Frozen linear probe on zone labels, using the corrected geometry and the
  Phase 0 CV harness.
- **The pretrained encoder's frozen probe must beat the same probe on an
  ImageNet-21k-initialised encoder by ≥0.02 macro-F1, CV mean.**
- If it does not, the pretext objective is not adding information. Do not proceed
  to Phase 2 on the strength of hope; escalate per §Escalation rung 3.

---

## Gate 2 — Phase 2, downstream zone classification

### 2.A — Beat the clean baseline
- 3-fold patient-grouped CV, ≥3 seeds, **using `zone_dataset.py::patient_split`
  and `run_patient_cv.py` verbatim** — not a reimplementation. A comparison run
  on a different harness is not a comparison.
- **CV macro-F1 mean must exceed the Gate 0.C baseline by ≥0.02, with
  non-overlapping ±1 std.**

### 2.B — Class-level honesty
- Per-class F1 and recall reported for every class.
- **A macro-F1 gain driven entirely by the majority class does not pass.**
  The positive-class recall must not regress relative to Gate 0.C.

---

## Escalation

When a gate fails, identify the rung and state it in `NEXT_STEPS.md`. Do not skip
rungs, and do not jump to rung 4 out of frustration.

**Rung 1 — Tune.**
*Trigger:* gate missed narrowly (within ~0.02) AND all mechanism health checks passed.
*Action:* learning rate, schedule, epochs, λ warmup, EMA momentum. One or two
targeted variants. **Not a sweep** — r01–r26 demonstrates that sweeping without a
mechanism hypothesis produces 26 uninterpretable runs.

**Rung 2 — Fix downstream design.**
*Trigger:* gate missed by a wide margin, but Phase 1 health checks (1.A, 1.B) passed.
*Action:* the representation is fine; the problem is in pooling, head, loss, or
label definition. Iterate in Phase 2 **without retraining the encoder**.

**Rung 3 — Change the objective.**
*Trigger:* Phase 1's own health checks failed — probe doesn't beat baseline, or
PCA shows no structure.
*Action:* the pretext task itself is wrong. This warrants literature work and
architecture redesign, not hyperparameters. Open a new investigation.

**Rung 4 — Structural ceiling.**
*Trigger:* repeated failure at the same gate across ≥3 *mechanistically distinct*
variants, with tight variance across folds.
*Action:* stop modelling. Treat as evidence of a data-level bound — label noise,
weak modality correlate, or insufficient N. Audit the data, not the model.
**This is exactly what r01–r26 was and was misread as a modelling problem for
26 runs.** The tight fold std (±0.012) was the tell.

---

## Kill criteria

Escalate to the PI, and consider reframing the project, if:

- Gate 0.B cannot be met after two independent geometry-transfer approaches.
  Zone labels cannot be reliably placed on FP → the per-zone framing is not viable
  and the task should be reformulated at image level.
- Gate 1.B fails and rung-3 redesign also fails → FA supervision does not transfer;
  the honest conclusion is that leakage is largely FA-modality-specific.
- Gate 2.A is met but positive-class recall stays below **0.50** → the model
  detects leakage too unreliably for the stated clinical screening motivation.

A clean negative result on any of these is publishable and is a legitimate
outcome. Do not tune until something looks good.
