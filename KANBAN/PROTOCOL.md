# KANBAN PROTOCOL

The experiment ledger for LARA. Authoritative for **what we empirically learned**.
Code and `docs/PHASE_*.md` remain authoritative for **what the system is**.

Do not put run narratives in architecture docs. Do not put implementation truth
(module shapes, defaults, hyperparameter meanings) in KANBAN.

---

## 1. Folder layout

```
KANBAN/
├── PROTOCOL.md                 # this file
├── PHASE_0_geometry/
│   ├── README.md               # index: investigation status + run table
│   └── investigation_NNN/
│       ├── DESCRIPTION.md
│       ├── OBSERVATIONS.md
│       ├── NEXT_STEPS.md
│       └── <run_slug>/
│           ├── DESCRIPTION.md
│           ├── OBSERVATIONS.md
│           ├── NEXT_STEPS.md
│           ├── PLAN.md            # optional
│           ├── GUIDE.md           # optional: exact launch commands
│           ├── METRIC_READOUT.md  # optional: facts only, pulled from W&B
│           └── ANALYSIS.md        # optional: interpretation of readout
├── PHASE_1_pretext/
├── PHASE_2_downstream/
└── PHASE_3_legacy/             # backfilled r01–r26 + patient-aggregation
```

## 2. The triad

Every investigation and every run has exactly three files. No exceptions, no
invented file types (no `HYPOTHESES.md`, no `RUNS.md`, no postmortems).

| File | Role |
|---|---|
| `DESCRIPTION.md` | The question / hypothesis, status, links to parents and baselines |
| `OBSERVATIONS.md` | Evidence, how beliefs changed, conclusions |
| `NEXT_STEPS.md` | Concrete next action, or `spawned investigation_NNN` |

Status values: `OPEN` / `RUNNING` / `PAUSED` / `CLOSED`.

## 3. Investigation vs run

| New **investigation** when… | New **run** when… |
|---|---|
| The scientific question changes | Same question, different config |
| A new mechanism family is being tested | Ablation, repeat, or sweep arm |
| A closed thread spawned a follow-up question | Hyperparameter tick or resume |

When unsure: add a run to the current investigation.

## 4. Lifecycle

Adapted from the standing research lifecycle. There is no per-run tech-lead
review on this project; step 5 is self-directed with batched PI reporting.

```
1. Hypothesis      → read PHASE README + prior OBSERVATIONS before proposing anything.
                     Do not revive a rejected idea without new evidence.
                     Understand WHY the change should work and WHAT changes in the pipeline.

2. Plan            → write PLAN.md + run DESCRIPTION.md BEFORE any code is written.
                     DESCRIPTION.md states the hypothesis and the config delta.

3. Implement       → agent reads the target code first, then applies changes.
                     Code must match PLAN.md. Update GUIDE.md if commands changed.

4. Launch          → follow GUIDE.md. Read every hyperparameter and justify each one.
                     Record the W&B run ID in DESCRIPTION.md immediately.

5. Analyse         → CONCURRENTLY: (a) agent pulls real metrics via W&B → METRIC_READOUT.md,
                     (b) you read the W&B graphs yourself and form an independent view.
                     Then reconcile the two into ANALYSIS.md. Disagreement is signal.

6. Record          → update run triad → parent investigation OBSERVATIONS/NEXT_STEPS
                     → PHASE README index. Then consult docs/ACCEPTANCE_GATES.md
                     §Escalation to decide what class of change comes next.
```

**PI reporting is batched, not per-run.** Generate the digest *from* the PHASE
README tables rather than writing it fresh. If the tables are current, the digest
is a copy-paste.

## 5. W&B is the metric ground truth

- W&B project: **`uveitis-per-zone`**
- Every paid science run passes `--require-wandb`. A missing or broken tracker
  must fail the run loudly rather than train silently.
- Run naming contract: `Investigation NN · axis · variant`
  (e.g. `Inv 03 · geometry-source · fa-crosshair-direct`)
- **Never hand-type a metric into KANBAN.** Metrics enter only via `run_history.py`
  pulling from the W&B API. If a number cannot be traced to a W&B run ID, it does
  not go in `METRIC_READOUT.md`.
- Known gap: the legacy 3-fold CV harness ran with `--no-wandb`, so the historical
  0.519 ± 0.012 traces only to `runs/protocol_cv3_r14_config/cv_summary.json`.
  All new CV runs must log to W&B.

## 6. Writing discipline

- Write less. New investigations are cheap; bloated files are expensive.
- **Never delete or rewrite history.** Wrong conclusions stay. Add a dated correction
  beneath them.
- Link, don't nest. Overlap via markdown links, not deeper folders.
- Load context lightly: `PHASE_*/README.md` → active investigation triad → current
  run folder. Never load the whole KANBAN.
- No secrets, no large log dumps. Link out.

## 7. Code hygiene

- Experimental knobs are CLI flags.
- Settled fixes get baked into defaults.
- Abandoned mechanisms get removed so the ledger matches the code.

## 8. Anti-patterns

- Reporting a single-split number as a result. **CV mean is the unit of evidence.**
  Run r26 hit ~0.58 val and cratered to ~0.45 test; single splits are not
  admissible for any gate decision.
- Inventing or estimating a metric.
- Starting a run before `DESCRIPTION.md` exists.
- Re-testing something already closed in `PHASE_3_legacy` without new evidence.
  **Note:** all PHASE_3 results were produced on corrupted zone geometry
  (see `docs/PHASE_0_geometry_rebuild.md`). They are void as evidence about
  modelling choices, but retained as a record of what was attempted.
