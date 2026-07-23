# PHASE 0 — Geometry rebuild & clean baseline

**Status:** blocking. No architecture work begins until Gate 0 passes.
**Gate:** `docs/ACCEPTANCE_GATES.md` §Gate 0 (sub-gates A, B, C, D)
**Ledger:** `KANBAN/PHASE_0_geometry/`

---

## 1. What is broken and why

Established by `docs/DATA_AUDIT.md` and `docs/GEOMETRY_AUDIT.md`.

**The zone crosshair is drawn on the FA, not the FP.** `pre_processing.py` calls
`detect_crosshair_from_yellow` on the FP, where no crosshair exists.

**`extract_zones.make_yellow_mask` is not selective on Optos FP.** Its predicate
requires `B<100`, `R−B>20`, `G−B>20`. Measured Optos FP blue-channel mean over
content is **1.1**, so the mask matches a **median 63% of FP retinal content**.
On FA the same mask matches **0.36%**.

Two independent failures follow:

| Failure | Mechanism | Measured |
|---|---|---|
| Wrong zone geometry | Hough runs on a mask covering 63% of the retina; finds spurious lines; the intersection usually lands in-bounds so `fovea_fallback` stays `False` | FP LOOSE: median 2315 horizontal vs 79.5 vertical lines; **0/60 images** show a balanced crosshair signature. FP↔FA fovea distance median **1442 px**. Only **4 of 692** pairs are within 200 px. Visual QC **0/25** correct. |
| Destroyed image content | `remove_yellow_overlay` inpaints that same 63% mask | Median **63%** of content pixels altered by >5; median PSNR **20.0 dB** |

**Consequence:** ~99% of the 7720 training rows carry zone labels attached to the
wrong retinal tissue, on images with most of their content inpainted away. The
legacy 0.519 ± 0.012 is a noise floor produced by this corruption, not a
modelling ceiling. Its ±0.012 tightness was the signature of a *constant*
corruption process — which is exactly what escalation rung 4 in
`ACCEPTANCE_GATES.md` describes, misread for 26 runs as a modelling problem.

**Why the fix is exact:** the zone constants reproduce the drawn overlay to the
pixel. `INNER_R_MM × PX_PER_MM = 3.0 × 53 = 159 px`;
`OUTER_R_MM × PX_PER_MM = 16.0 × 53 = 848 px`. Independent measurement of a
sample FA overlay gives r ≈ 161 and r = 848. Feeding `make_zone_mask` the FA's
detected `(cx, cy, angle_deg)` reproduces the grader's own zone boundaries.

---

## 2. Objective

Rebuild zone geometry from the FA crosshair, transfer it into FP image space,
regenerate the training table, and re-establish a trustworthy baseline.

**Do not modify `extract_zones.py::make_zone_mask` geometry semantics.** It is
correct. Only its *inputs* were wrong.

---

## 3. Step 1 — Stop inpainting the FP, and make the change reversible

All Phase 0 changes are gated behind a new flag on `pre_processing.py`:

```
--geometry-source {fp_crosshair, fa_crosshair}    # default: fa_crosshair
```

`fp_crosshair` preserves the existing behaviour exactly. This means the old
pipeline stays runnable and reproducible without git gymnastics, and the two
paths can be compared directly on the same machine.

Output goes to a **new** directory. Do not overwrite `processed_image_arrays/` or
`processed_image_arrays_multiclass/` — they are evidence.

### Stop inpainting

The FP carries no overlay. Inpainting it is pure damage.

- Under `fa_crosshair`, write the **raw** (OD-standardized) FP array to cache.
  `remove_yellow_overlay` is not called on the FP at all.
- Retain a safety check: run TIGHT-mask detection
  (`(R>110)&(G>110)&(B<90)&((R−B)>60)&((G−B)>60)`) on each FP. If it yields a
  balanced crosshair (both H and V counts in `[1, 15]`), **flag the image for
  manual review** — do not auto-inpaint.
- Log the count of flagged images. Expected: near zero.

Gate 0.D requires median PSNR(raw, cached) **> 40 dB**.

## 4. Step 2 — Extract geometry from the FA

For every visit-eye with an `_FA_0001` (RGB) file:

1. Apply the **same OD/OS standardization flip** used for the FP, so FA and FP
   land in a common convention. This must be verified, not assumed — write a test.
2. Detect the crosshair with the **TIGHT** mask using the existing
   `detect_crosshair_from_yellow` logic. One acceptance-criterion addition:
   - Require a **balanced H/V signature** (both counts in `[1, 15]`) as an
     acceptance condition, not merely a successful in-bounds intersection. This is
     what distinguishes a real crosshair from Hough noise, and its absence is why
     the FP failures were silent.

   *Note: no change to `line_intersection` is required. An earlier draft of this
   spec claimed int32 overflow at 4000 px; that was incorrect. The caller casts via
   `map(int, item)`, giving Python unbounded ints. The overflow affected an external
   verification script only. Retracted.*
3. Record `(cx_fa, cy_fa, angle_fa)` and the H/V counts per visit-eye.

Expected success: **≥95%** (audited 768/789 = 97.3%).

**Visit-eyes with no usable FA geometry are dropped from the training table.**
There is no fallback. The centre-fallback path must be deleted, not repaired —
its existence is why the corruption was invisible for 26 runs.

## 5. Step 3 — Transfer FA geometry into FP space

This is the one genuinely open technical problem in Phase 0. Implement **Approach
A first**; escalate only if it fails Gate 0.B.

### Approach A — direct transfer (implement first)

Assume FA and FP are co-registered (same device, same session) and apply
`(cx_fa, cy_fa, angle_fa)` directly as the FP's zone geometry.

Justification: the detected FA crosshair centres cluster tightly near the frame
centre (median ≈ (1980, 1960) in a 4000×4000 frame) because the operator centres
the fovea at capture. Audited FA→FP content-mask offset is median 86 px.

Risk to quantify: 86 px against an inner-zone radius of 159 px is a ~54%
relative error **for the four inner zones**. The ring and periphery zones are far
more tolerant. Report Gate 0.B containment **stratified by zone group** so this
is visible rather than averaged away.

### Approach B — per-pair rigid refinement (escalate here first)

Estimate a per-pair translation (and small rotation/scale) by maximising
correlation between the FA intensity and the **negated FP green channel** —
vessels are bright in FA and dark in FP; audited correlation is negative, which
confirms the sign. Apply the transform to `(cx_fa, cy_fa, angle_fa)`.

Audited caveat: mask-FFT translation alone gave median IoU 0.827 with several
catastrophic outliers (one pair at −1168 px). A vessel-based objective should do
better than the mask-based one that produced those numbers, but any pair whose
refinement *worsens* content-mask IoU must fall back to Approach A and be flagged.

### Approach C — anatomical FP landmarks (escalate second)

Detect the optic disc on the FP directly (brightest coherent structure) and
derive the fovea as `OD − rotate((ONH_OFFSET_X, ONH_OFFSET_Y), angle_fa)`, taking
the angle from the FA. Fully independent of registration.

## 6. Step 4 — Verify (do not assume) the ONH placement convention

`make_masks` rotates the quadrant boundaries by `angle_deg` — the `rx`, `ry`
coordinates are rotated — but places the ONH ellipse at an **unrotated** offset:

```python
onh_cx = cx + onh_offset_x
onh_cy = cy + onh_offset_y
```

**This is a discrepancy to investigate, not a confirmed bug.** Whether it is
wrong depends entirely on what the annotation tool did:

- If the tool drew the disc marker at a fixed unrotated image-space offset from
  the crosshair centre, **the current code is correct and must not be changed.**
- If the tool rotated the disc marker with the crosshair, the current code
  displaces zone 9 by `onh_offset_x · sin(θ)`. At 270 px and θ = −25°, that is
  ~114 px against ellipse radii of `rx=80, ry=95` — enough to miss the disc.

**How to resolve it empirically, without needing the tool's documentation:**
run the Gate 0.B optic-disc containment test *both ways* — unrotated offset and
rotated offset — on the same corrected geometry, and compare containment rates.
The convention that matches the annotation tool will show a clearly higher rate.
Report both numbers. Whichever wins becomes the documented convention.

Restrict any change to this single expression. **`extract_zones.py` is otherwise
correct and stays locked.** Its zone geometry reproduces the drawn overlay exactly
(`INNER_R_MM × PX_PER_MM = 159` vs measured ≈161; `OUTER_R_MM × PX_PER_MM = 848`
vs measured 848), and `detect_crosshair_from_yellow` succeeds on 97.3% of FAs.
The pipeline failure was in `pre_processing.py` handing it the wrong image, not in
this file.

## 6b. Step 4b — Verify zone-index and laterality correspondence

Unverified assumption, invisible until now because the geometry was corrupt:
**does spreadsheet `Zone{N}_label` refer to the same retinal region as
`make_zone_mask(..., zone_number=N, ...)`?**

If the annotation tool numbered zones in a different order, every label is
attached to the wrong zone — a pure permutation error that no amount of modelling
can recover from and that would be extremely hard to diagnose later.

The same question applies to laterality. `pre_processing.py` flips OS eyes to an
OD-axis convention. If the clinician graded OS eyes in their native orientation,
then `Zone3` on an OS eye denotes different tissue post-flip than `Zone3` on an
OD eye, and the flip must be accompanied by a corresponding zone-index permutation
(typically a nasal↔temporal swap of the quadrant indices).

**Required actions:**
1. Locate documentation for the annotation protocol or grading tool. Record the
   zone numbering convention in `KANBAN/PHASE_0_geometry/`. If no documentation
   exists, this is a single question for the PI — ask once and record the answer.
2. Confirm whether the grader worked in native eye orientation or in a
   standardized orientation.
3. Add an assertion in the dataset layer that `ZONE_NAMES[N]` and the spreadsheet
   column `Zone{N}_label` are documented as corresponding, so the assumption is
   recorded in code rather than implicit.

**Weak supporting evidence, not a substitute for confirmation:** the audited
per-zone label distributions are structured rather than random — zones 1–4 cluster
at ~18–19% tier-1, zones 5–10 at ~11–15% — which is consistent with a coherent
numbering scheme. This is suggestive only.

This is a verification item, not a blocker. Do not hold Phase 0 for it, but do not
report Gate 0.C results as trustworthy until it is resolved.

## 7. Step 5 — Regenerate the training table

Emit both variants, as the current pipeline does:
- binary (`{1,2} → 1`), and
- multiclass (raw `0/1/2`) via `--multiclass-zone-labels`

Each row must additionally carry: `geometry_source` (`fa_direct` / `fa_refined` /
`fa_anatomical`), `hv_counts`, `od_containment_pass` (bool), and
`transfer_residual_px` where Approach B ran.

Report the new row count against the previous 7720 and account for every dropped
row.

## 8. Step 6 — Validation (Gate 0.B)

**Automated — optic-disc containment.** Detect the OD centroid on each FP
independently of the zone geometry. Test whether it falls inside the projected
zone-9 ellipse. Required: **≥85%**. Report overall and stratified by zone group
(inner / ring / ONH / periphery).

This test is strong precisely because it uses an *independent* signal: the FP's
own brightest structure versus a geometry derived entirely from the FA.

**Human — visual QC.** Render 25 random visit-eyes: FP with all 10 zone
boundaries and the transferred fovea marked, beside the FA with its detected
crosshair. Required: **≥23/25 judged correct by a human.** Current pipeline
scores 0/25. This gate cannot be automated away.

## 9. Step 7 — Establish the real baseline (Gate 0.C)

Re-run the exact legacy r14 configuration on the regenerated data:

```bash
python run_patient_cv.py \
  --folds 3 --seed 13 \
  --csv processed_image_arrays_multiclass_v2/zone_training_table.csv \
  --data-root processed_image_arrays_multiclass_v2 \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --epochs 20 --batch-size 32 --lr 1e-4 \
  --weight-decay 1e-4 --image-size 288 --zone-embed-dim 64 \
  --head-hidden 256 --patience 8 --grad-clip-norm 1.0 \
  --output-dir runs/phase0_baseline_r14_corrected \
  --wandb-project uveitis-per-zone --require-wandb
```

The harness previously ran with `--no-wandb`; it must now log. Add
`--require-wandb` support to `run_patient_cv.py` if absent.

Record `test_macro_f1_mean ± std`, per-class recall, and the W&B run IDs in
`KANBAN/PHASE_0_geometry/`. **This number is the project baseline.**

Run the identical config on the **old** table as a control, so the delta
attributable to the geometry fix is measured rather than assumed.

---

## 10. Deliverables

1. `docs/geometry_v2/` — containment statistics, transfer residuals, dropped-row accounting
2. `docs/geometry_v2/qc/` — the 25 side-by-side QC renders
3. Regenerated binary + multiclass tables under a **new** output directory
   (do not overwrite `processed_image_arrays/` — the old data is evidence)
4. `KANBAN/PHASE_0_geometry/investigation_001/` triad, populated
5. Two W&B CV runs: corrected-data baseline and old-data control

## 11. What Phase 0 might make unnecessary

State this plainly in the Gate 0.C analysis: if the corrected baseline lands far
above 0.519, a large share of the project's difficulty was infrastructure, not
architecture. Phase 1 must then justify itself against the **new** number, not
the old one.

It remains entirely possible that corrected geometry alone closes most of the
gap and the SSL work becomes a refinement rather than a rescue. That is a good
outcome, and it must not be obscured by proceeding to Phase 1 on momentum.
