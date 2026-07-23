# AGENTS.md — entry point for coding agents (Claude Code, Cursor, Codex, etc.)

Read this first. It tells you where the authoritative context lives, how to run things, and the
**one workflow convention that is mandatory in this repo**: how investigations and experiments are
recorded under `KANBAN/`.

---

## What this project is (30-second version)

Predict **per-zone leakage severity** of retinal inflammation (uveitis) from **standard fundus
photographs**, using labels a clinician graded on the invasive gold standard (fluorescein
angiography). Each eye has **10 anatomical zones**, each graded tier **0 / 1 / 2**, collapsed to
**binary** (`0 → 0`, `{1,2} → 1`) at the zone level. FA never enters the network — it only produced
the labels. Evaluation is **patient-level** (zones from one eye must not leak across splits). The
scientific contribution is **per-zone localization**, not just "does this patient have uveitis" —
keep that in mind for every modeling decision.

For the honest, evidence-backed state of the project, read these two documents **before doing any
modeling work** — do not re-derive their conclusions:

- **`KANBAN/00_SITUATION_AND_CONTEXT.md`** — what we built vs. what we pitched, the 26 prior runs, the
  0.519 ± 0.012 CV ceiling, and the known confounds.
- **`KANBAN/01_NEXT_STEP_OPTIONS.md`** — the committed decision analysis that chose the next
  architecture (Option A) and ruled the alternatives in/out.

---

## The KANBAN convention (MANDATORY — this is why this file exists)

**`KANBAN/` is a living, authoritative reference.** It is the shared long-term memory for humans and
AI agents: the place to understand what has been tried, what direction we committed to and why, and
how to run each experiment. Treat it as the source of truth. Keep it current.

Structure:

```
KANBAN/
├── 00_SITUATION_AND_CONTEXT.md      # project state of play (read first)
├── 01_NEXT_STEP_OPTIONS.md          # committed decision analysis (read second)
└── INVESTIGATIONS/
    └── <NN>_<slug>/                 # one INVESTIGATION = one committed research direction
        ├── DESCRIPTION.md           # what/why/how + expected results + success criterion
        ├── GUIDE.md                 # exact runbook to execute the experiment end to end
        └── <experiment-slug>/       # OPTIONAL sub-folders, one per concrete experiment/variant
            └── RESULTS.md
```

**The rule:** whenever the user and the AI assistant have **agreed to go forward with a specific
investigation**, that agreement must be recorded here before/as the work starts:

1. **New direction →** create a new folder under `KANBAN/INVESTIGATIONS/` named `NN_<slug>` (next
   number, descriptive slug), and add a thorough **`DESCRIPTION.md`** (technical + implementation +
   conceptual detail: what we're testing/changing, why, and the expected result / success criterion)
   and a **`GUIDE.md`** (the exact, accurate, end-to-end commands to run it).
2. **New experiment within an existing direction →** create a new **experiment sub-folder** inside the
   relevant existing investigation folder (e.g. a resolution sweep, or an added model head), with its
   own notes/results.

Which of the two applies depends on what the user wants — **ask if it's ambiguous.** Do not start a
committed experiment without recording it here.

**Quality bar for these docs:** thorough, end-to-end, and **accurate** — an agent must be able to pick
up the work cold from these files alone (to answer questions, resume, or start a new experiment).
Cross-link related KANBAN docs. When an experiment finishes, **write the result back** (CV numbers with
mean ± std, W&B run URLs, pass/fail against the success criterion) into the investigation's
`DESCRIPTION.md` or a `RESULTS.md`, and update `RESEARCH_LOG.md` if it belongs in protocol history.

Current investigations:
- **`INVESTIGATIONS/01_full-image-masked-pooling-cross-zone-attention/`** — Option A: full-image
  backbone → masked zone pooling → cross-zone attention → 10 per-zone logits. **Active / building.**

---

## Environment & how to run

- **Python interpreter (use directly, don't rely on `conda activate`):**
  `/home/shashwat/miniconda3/envs/venv/bin/python` — deps in `requirements.txt` are installed here
  (torch, torchvision, timm, open_clip_torch, scikit-learn, opencv-python, **wandb**).
- **GPUs:** 4 available — 2× RTX A6000 (49 GB, GPU 0 & 3) and 2× RTX 3090 (24 GB, GPU 1 & 2). Pin one
  with `export CUDA_VISIBLE_DEVICES=<n>`. Also `export MKL_THREADING_LAYER=GNU` for the CV runner.
- **Data (use the multiclass CSV/root — it keeps raw tiers 0/1/2):**
  `processed_image_arrays_multiclass/zone_training_table.csv` with data-root
  `processed_image_arrays_multiclass`.
- **Splits:** patient-level. `splits/canonical_split.json` is the **stale** seed-13 91-patient split
  (only ~69% of patients; being replaced in Phase 0 — see investigation 01). Use CV, not single-split,
  for any real comparison.

### Key code files

| File | Role |
|------|------|
| `train_convnext.py` | Main trainer (crop baseline): ConvNeXt-Tiny + zone-id embedding, focal/CE loss, W&B logging, early stop on val macro-F1. |
| `train_clip_convnext.py`, `train_retfound.py` | Alternative backbones (both lost to ConvNeXt-Tiny at this data scale). |
| `zone_dataset.py` | `ZoneImageDataset`, patient-split logic, `load_zone_records`. |
| `extract_zones.py` | Zone geometry / masks (`make_zone_mask`), crosshair/fovea detection, `apply_zone_and_crop` (the crop path). |
| `losses.py` | CE + focal loss (`build_loss`). |
| `run_patient_cv.py` | K-fold patient-level CV harness (currently 3-fold). |
| `pre_processing.py` | Raw data → cleaned arrays + `zone_training_table.csv` (+ fovea sidecars). |
| `RESEARCH_LOG.md` | Protocol run history (r01–r26). |

---

## Experiment logging (W&B is the primary logger)

Training logs to **Weights & Biases**, project **`uveitis-per-zone`**
(`https://wandb.ai/smahalanobis-uc-davis/uveitis-per-zone`), on by default; `--no-wandb` disables it.
Credentials are in `~/.netrc`. `train_convnext.py` and `train_clip_convnext.py` log full per-epoch
train/val metrics, test metrics, and per-class ROC curves. Notes: `train_retfound.py` only logs a
couple of end-of-run test scalars, and `run_patient_cv.py` runs folds with `--no-wandb` (fold results
live in `metrics.json` / `cv_summary.json`). Every run also writes `runs/<name>/{metrics.json,
train.log, best.pt}` locally.

---

## Working norms

- **The only valid comparison is CV-to-CV.** Single-split test F1 has produced false signals (r14
  0.56 vs CV 0.52; r26 val 0.58 vs test 0.44). Never claim a win from one split.
- **Don't relitigate settled decisions.** Kept-because-validated: exclude-tier1 in training, focal
  γ=3 + inverse class weights, ConvNeXt-Tiny backbone. See `01_NEXT_STEP_OPTIONS.md`.
- **Report honestly.** If a result fails the pre-declared success criterion, say so — a clean negative
  is a real finding here (it points to data/label quality, not architecture).
- **Keep KANBAN current** as described above. It's the contract that lets the next agent continue.
