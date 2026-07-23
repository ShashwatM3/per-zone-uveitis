# PROMPT 02 — Phase 0: geometry rebuild & clean baseline

Send after PROMPT 01 has completed and been reviewed.

**Read `docs/PHASE_0_geometry_rebuild.md` and `docs/ACCEPTANCE_GATES.md` §Gate 0
in full before writing any code.** This prompt is a sequencing and guardrail
document; the spec is authoritative for content.

Stay on the current branch. Do not create `feature/xjepa`.

---

## Ground rules

1. **`extract_zones.py` stays locked**, with exactly one possible exception
   (the ONH offset convention, §6 of the spec — and only after the containment
   test tells you which convention is right). Its zone geometry is verified
   correct. The pipeline failure was in `pre_processing.py` passing it the wrong
   image.
2. **Nothing is overwritten.** All output goes to a new directory. The existing
   `processed_image_arrays/` and `processed_image_arrays_multiclass/` are evidence
   and must remain byte-identical.
3. **The old path stays runnable** behind `--geometry-source fp_crosshair`.
4. **Stop and report at each checkpoint below.** Do not run the full sequence
   unattended — checkpoint C requires human visual review that cannot be automated.

---

## Step 1 — `--geometry-source` flag and FP inpainting removal

Add to `pre_processing.py`:

```
--geometry-source {fp_crosshair, fa_crosshair}   # default fa_crosshair
--output-root <path>                              # required; must not be an existing dir
```

Under `fa_crosshair`:
- Do **not** call `remove_yellow_overlay` on the FP. Write the raw
  OD-standardized array.
- Run TIGHT-mask detection on each FP as a safety check. TIGHT is
  `(R>110)&(G>110)&(B<90)&((R−B)>60)&((G−B)>60)`. If it produces a balanced
  crosshair (both H and V counts in `[1,15]`), flag that image for manual review.
  Do not auto-inpaint. Report the flagged count.

Under `fp_crosshair`: behaviour is unchanged from today.

**→ CHECKPOINT A.** Report: flagged-FP count, and median PSNR between raw FP and
the newly cached array over 30 samples. Gate 0.D requires **> 40 dB**. Stop.

---

## Step 2 — Geometry from the FA

Implement in `lara/geometry/`.

For each visit-eye with an `_FA_0001` file:
1. Apply the **same OD/OS standardization flip** used for the FP. Write a unit
   test asserting FA and FP land in the same convention. Do not assume this.
2. Run `detect_crosshair_from_yellow` with the **TIGHT** mask.
3. Accept only if the H/V signature is **balanced** — both counts in `[1,15]`.
   A successful in-bounds intersection alone is not sufficient; that permissiveness
   is exactly why the FP failures were silent.
4. Record `(cx_fa, cy_fa, angle_fa)`, H/V counts, and the acceptance decision.

Expected success ≥95% (audited: 768/789 = 97.3%).

**Visit-eyes with no accepted FA geometry are dropped from the training table.**
Delete the centre-fallback path entirely — do not repair it. Its existence is why
the corruption stayed invisible across 26 runs.

**→ CHECKPOINT B.** Report: acceptance rate, failure reasons with counts, and how
many visit-eyes will be dropped. Stop.

---

## Step 3 — Transfer FA geometry to FP space

Implement **Approach A only** for now (spec §5).

Apply `(cx_fa, cy_fa, angle_fa)` directly as the FP's zone geometry. Record a
per-pair `transfer_method` field so B and C can be added later without ambiguity.

Do not implement Approaches B or C yet. They are escalation paths, invoked only if
Gate 0.B fails.

---

## Step 4 — Validation (Gate 0.B) — the decisive step

**Optic-disc containment.** Detect the OD centroid on each FP *independently* of
the zone geometry (brightest coherent structure). Test whether it falls inside the
projected zone-9 ellipse (`rx=80, ry=95`).

Run this **twice**, once per ONH offset convention:
- unrotated: `onh_cx = cx + onh_offset_x` (current code)
- rotated: `onh_cx = cx + onh_offset_x·cos θ − onh_offset_y·sin θ`, similarly for y

Report both containment rates. The higher one identifies the annotation tool's
convention. Document the winner; only then consider touching `extract_zones.py`.

Report containment **stratified by zone group** (inner / ring / ONH / periphery).
A single averaged number can hide "outer zones fine, inner zones broken", which is
the expected failure mode given an ~86 px transfer error against a 159 px inner
radius.

**Visual QC.** Render 25 random visit-eyes to `docs/geometry_v2/qc/`: FP with all
10 zone boundaries and the transferred fovea marked, beside the FA with its
detected crosshair, both downsampled to ~1200 px.

**→ CHECKPOINT C — HUMAN REVIEW REQUIRED.** Report containment numbers (both
conventions, stratified) and list the 25 QC filenames. **Stop and wait.** Gate 0.B
requires ≥85% containment and ≥23/25 human-judged correct. Do not proceed on
automated numbers alone.

---

## Step 5 — Zone-index and laterality verification

Per spec §6b. This is a documentation task, not code.

Search the repo, the annotation workbooks, and any protocol docs for the zone
numbering convention. Determine whether spreadsheet `Zone{N}_label` corresponds to
`ZONE_NAMES[N]` in `extract_zones.py`, and whether the grader worked in native or
standardized eye orientation.

If it cannot be determined from available material, **say so explicitly** and list
it as a question for the PI. Do not guess, and do not let it block Steps 1–4.

Add an assertion in the dataset layer recording the assumed correspondence, so it
lives in code rather than implicitly.

---

## Step 6 — Regenerate the training table

Emit both variants: binary (`{1,2}→1`) and multiclass (raw `0/1/2`).

Each row additionally carries: `geometry_source`, `transfer_method`, `hv_counts`,
`od_containment_pass`, `onh_convention`.

Report the new row count against the previous 7720 and **account for every dropped
row**.

Handle the 84 images at 3900×3072 (patients 011–014, 016, 029, 039–040, 048,
053–055, 070): crop to content bbox before any resize, and assert post-crop aspect
ratio is within 10% of 1:1, flagging otherwise.

---

## Step 7 — Establish the real baseline (Gate 0.C)

Add `--require-wandb` to `run_patient_cv.py` if absent (the legacy harness passed
`--no-wandb`, which is why the 0.519 figure has no W&B trace).

Run the **exact** r14 configuration on the regenerated data:

```bash
python run_patient_cv.py \
  --folds 3 --seed 13 \
  --csv <new_output_root>_multiclass/zone_training_table.csv \
  --data-root <new_output_root>_multiclass \
  --loss focal --focal-gamma 3.0 --class-weighting inverse \
  --exclude-tier1 --epochs 20 --batch-size 32 --lr 1e-4 \
  --weight-decay 1e-4 --image-size 288 --zone-embed-dim 64 \
  --head-hidden 256 --patience 8 --grad-clip-norm 1.0 \
  --output-dir runs/phase0_baseline_r14_corrected \
  --wandb-project uveitis-per-zone --require-wandb
```

Then run the **identical config on the old table** as a control, so the delta
attributable to the geometry fix is measured rather than assumed.

Record both results, per-class recall, and all W&B run IDs in
`KANBAN/PHASE_0_geometry/investigation_001_geometry_rebuild/`.

**→ CHECKPOINT D.** Report both CV results side by side.

---

## KANBAN discipline

Before Step 1: create the run folder
`KANBAN/PHASE_0_geometry/investigation_001_geometry_rebuild/run_001_fa_geometry/`
with `DESCRIPTION.md` stating the hypothesis and planned config delta. Per
`PROTOCOL.md`, this is written **before** implementation, not after.

After Checkpoint D: fill `OBSERVATIONS.md` (metrics pulled from W&B, never
hand-typed) and `NEXT_STEPS.md`, then update the phase README tables.

---

## Report format

At each checkpoint, report only: the numbers requested, anything that contradicts
the spec's stated expectations, and anything you had to decide that the spec did
not cover. Do not summarise what you implemented — the diff shows that.
