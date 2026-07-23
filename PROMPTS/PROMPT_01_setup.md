# PROMPT 01 — Setup: docs, KANBAN scaffold, legacy backfill

Send this to the in-codebase coding agent **first**.

This prompt performs **repository organisation only**. Do not modify any pipeline
code, do not run training, do not implement anything from the phase specs. If you
find bugs while reading, record them in the relevant `OBSERVATIONS.md` — do not
fix them.

Work through the four tasks in order. Report after each.

---

## Preconditions

The following files have been placed in the working directory by the user:

```
docs/PHASE_0_geometry_rebuild.md
docs/PHASE_1_pretext.md
docs/PHASE_2_downstream.md
docs/ACCEPTANCE_GATES.md
docs/AUGMENTATION_POLICY.md
KANBAN/PROTOCOL.md
```

Read `KANBAN/PROTOCOL.md` and `docs/ACCEPTANCE_GATES.md` in full before doing
anything else. Everything below assumes you have.

**Authority rule.** `docs/PHASE_0_geometry_rebuild.md`, `docs/PHASE_1_pretext.md`,
`docs/PHASE_2_downstream.md`, `docs/ACCEPTANCE_GATES.md` and
`docs/AUGMENTATION_POLICY.md` are authoritative. If any pre-existing document in
the repo contradicts them, **the new files win**. Report the contradiction
explicitly; do not silently reconcile it, and do not follow the older document.

---

## TASK 0 — Inventory. Change nothing.

Report all of the following before doing anything else:

1. `git branch --show-current` and `git log -1 --oneline`
2. `git status --short` — the complete list, including untracked files
3. `find KANBAN -type f 2>/dev/null` — a `KANBAN/` directory already exists and was
   **not** created by any prior instruction. Report its full contents verbatim.
   **Do not delete, move, or overwrite anything already inside it.**
4. `ls -la docs/` — report every entry
5. For `docs/PROJECT_CONTEXT.md` and any file in `docs/` whose name references
   architecture or model strategy: print the first 40 lines of each. These predate
   the new phase specs and may contain superseded guidance.

Then **stop and report**. Do not begin Task 1 until the inventory has been reviewed.

---

## TASK 1 — Stay on the current branch

1. Record the current branch name and HEAD commit hash. Report both.
2. **Do not create any new branch.** Phase 0 is a data-pipeline correction that is
   valuable regardless of which architecture wins, so it lands on the current
   branch and propagates to everything downstream. `feature/xjepa` is branched
   later, only once Gate 0 has passed.
3. Commit the six supplied files here:
   `docs: add phase specs, acceptance gates, augmentation policy, KANBAN protocol`

Safety for Phase 0 is handled by a `--geometry-source {fp_crosshair, fa_crosshair}`
flag and a new output directory, not by branching. Nothing existing gets
overwritten.

## TASK 2 — KANBAN scaffold

A `KANBAN/` directory already exists with at least one investigation in it,
created outside these instructions. **Reconcile, do not replace:**

- If the existing investigation covers the same question as
  `investigation_001_geometry_rebuild` below, keep the existing folder and bring it
  up to the triad standard rather than creating a duplicate. Report what you found
  and what you changed.
- If it covers a different question, leave it untouched and place it in the
  appropriate `PHASE_*/README.md` investigation table.
- If it has files outside the triad (any name other than `DESCRIPTION.md`,
  `OBSERVATIONS.md`, `NEXT_STEPS.md`, or the optional per-run files listed in
  `PROTOCOL.md`), **report them — do not delete them.**

Create the following, obeying the triad rule in `PROTOCOL.md` (exactly three files
per investigation and per run — no invented file types):

```
KANBAN/
├── PROTOCOL.md                     # already present
├── PHASE_0_geometry/
│   ├── README.md
│   └── investigation_001_geometry_rebuild/
│       ├── DESCRIPTION.md
│       ├── OBSERVATIONS.md
│       └── NEXT_STEPS.md
├── PHASE_1_pretext/README.md
├── PHASE_2_downstream/README.md
└── PHASE_3_legacy/
    ├── README.md
    └── investigation_001_perzone_crop/
        ├── DESCRIPTION.md
        ├── OBSERVATIONS.md
        ├── NEXT_STEPS.md
        └── <one folder per legacy run — Task 3>
```

**Phase README format** — two tables, nothing else:

Investigations: `| # | slug | status | one-line conclusion |`
Runs: `| run # | slug | W&B ID | state | investigation | verdict |`

For `PHASE_1_pretext` and `PHASE_2_downstream`, create the README with empty
tables plus a one-paragraph purpose statement drawn from the corresponding
`docs/PHASE_*.md`. Do not invent investigations.

For `investigation_001_geometry_rebuild`:
- `DESCRIPTION.md` — question: *"Can zone geometry be correctly recovered from the
  FA crosshair and transferred into FP image space?"*; status `OPEN`; link to
  `docs/PHASE_0_geometry_rebuild.md`; link to the gate it targets
  (`ACCEPTANCE_GATES.md` §Gate 0).
- `OBSERVATIONS.md` — placeholder.
- `NEXT_STEPS.md` — the first concrete action: Phase 0 Step 1.

---

## TASK 3 — Backfill legacy history

Reconstruct r01–r26 and the `patient-aggregation` experiment as a **CLOSED**
investigation under `PHASE_3_legacy/`.

### Sources, in priority order

1. `RESEARCH_LOG.md`
2. `runs/*/metrics.json` — contains exact args per run
3. `runs/protocol_cv3_r14_config/cv_summary.json` and its `fold*/metrics.json`
4. `WANDB_RUNS.md`
5. Branch `origin/feature/patient-aggregation` — read its README/log

**Never invent a metric.** If a number is not traceable to one of these sources,
write `UNKNOWN — not traceable`. Known-missing cases: the single-split r14 W&B ID
is unrecoverable (`train.log` absent, local `wandb/run-*` pruned), and the 3-fold
CV harness ran with `--no-wandb` so has no W&B IDs at all.

### Per-run folders

Name them `r<NN>_<short_slug>` (e.g. `r14_focal_g3_exclude_tier1`). Triad only:

- `DESCRIPTION.md` — what varied vs the previous run (one paragraph), the exact
  args from `metrics.json`, and the W&B ID or `UNKNOWN — not traceable`.
- `OBSERVATIONS.md` — the metrics actually recorded, **with the file path they came
  from**, plus a one-line verdict (`no movement` / `small gain` / `class collapse`
  / `val-test divergence`).
- `NEXT_STEPS.md` — `CLOSED — superseded by Phase 0 geometry rebuild`.

Two to six sentences each. This is a ledger, not a narrative.

### Investigation-level triad

`DESCRIPTION.md`: question — does per-zone crop classification with a strong
backbone recover FA-derived zone severity from FP? Status `CLOSED`. Record the
shipped design: black-masked pie-slice crops with a zone-id embedding, 10 zones
treated as i.i.d. samples — which diverged from the originally proposed
full-image masked pooling.

`OBSERVATIONS.md` must record, factually:
- Backbones swept: ConvNeXt-Tiny (best), CLIP ConvNeXt-L, EfficientNet-B3, RETFound
- Knobs swept r01–r26: focal loss, soft labels, exclude-tier1, oversampling, TTA, MixUp
- Best single split: r14, 0.560 test macro-F1
- Honest 3-fold patient CV of that config: **0.519 ± 0.012**, per-fold
  `[0.5323, 0.5223, 0.5032]`, source `runs/protocol_cv3_r14_config/cv_summary.json`
- r26: best validation (~0.58) then test collapse (~0.45)
- `patient-aggregation` (freeze r14 zone features → patient-level aggregator) was
  ceiling-capped

Then append this dated correction block verbatim, per the protocol's
never-rewrite-history rule:

```markdown
### CORRECTION — <today's date>

All results in this investigation were produced on corrupted zone geometry.
`pre_processing.py` detected the zone crosshair on the FP, where no crosshair
exists; the crosshair is drawn on the FA. `make_yellow_mask` matched a median 63%
of Optos FP retinal content (blue-channel mean 1.1), so Hough line detection ran
on noise. Measured FP↔FA fovea distance: median 1442 px; only 4 of 692 pairs
within 200 px; visual QC 0/25 correct. `remove_yellow_overlay` additionally
inpainted that same 63% mask (median PSNR 20.0 dB).

These runs are **void as evidence about modelling choices**. The 0.519 ± 0.012
figure is a noise floor produced by a constant corruption process, not a
modelling ceiling — its tight fold variance was the tell.

Retained as a record of what was attempted. Do not cite any conclusion from this
investigation as grounds for rejecting a modelling idea.
Sources: `docs/DATA_AUDIT.md`, `docs/GEOMETRY_AUDIT.md`.
```

`NEXT_STEPS.md`: `CLOSED — spawned PHASE_0_geometry/investigation_001_geometry_rebuild`

---

## TASK 4 — Scaffold the new package (empty)

Create the directory skeleton only. **No implementation.**

```
lara/
├── __init__.py
├── geometry/__init__.py      # Phase 0: FA crosshair detection, FA→FP transfer
├── data/__init__.py          # Phase 0/1: pair dataset, preprocessing v2
├── models/__init__.py        # Phase 1: encoder, predictor, zone head
├── losses/__init__.py        # Phase 1/2: dense latent loss, CORN ordinal
└── eval/__init__.py          # Phase 0/2: metrics, CV wrappers
```

**Do NOT archive or move `train_convnext.py`, `train_clip_convnext.py`,
`train_retfound.py`, or `run_patient_cv.py`.** Phase 0 Gate 0.C requires
`train_convnext.py` and `run_patient_cv.py` to run the r14 config on regenerated
data. Archiving happens at the start of Phase 1, not now.

Commit: `chore: scaffold lara package, KANBAN ledger, backfill legacy runs`

---

## Report back

1. Current branch name and HEAD hash (no new branch should have been created)
2. The KANBAN tree you created
3. How many legacy runs backfilled; how many had traceable metrics vs `UNKNOWN`
4. Any run directory in `runs/` **not** mentioned in `RESEARCH_LOG.md`
5. Anything in the sources that **contradicts** the summary above — report the
   contradiction, do not silently reconcile it

Then stop. Wait for PROMPT 02.
