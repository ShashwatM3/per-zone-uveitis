# Phase 0 — Results Note

**Date:** 2026-07-10
**Scope of this note:** items **3.1 (P0.4)** and **3.3 (P0.3)** from `GUIDE.md §3`.
Items 3.2 (156-patient split) and 3.4 (5-fold CV harness) are **not yet done**.

---

## 3.1 (P0.4) — `batch_to_device` unpack fix — ✅ DONE

- **Fixed:** `eval_convnext_sweep.py:116` — now unpacks 4 values
  (`images, labels, zones, _`) to match `train_convnext.batch_to_device`, which returns
  `(images, labels, zones, sample_weight)` since commit `ee22b6f`. It previously unpacked 3
  and would `ValueError` on any run.
- **Investigated but intentionally left alone:** `threshold_sweep.py:66` and
  `train_clip_convnext.py:416` also do a 3-value unpack, but both use a **local, 3-returning**
  `batch_to_device` defined in `train_clip_convnext.py` (line 328) — they were never broken.
  The GUIDE named the correct and only genuinely-broken file.
- **Verified:** all four call sites compile; `eval_convnext_sweep.py` is the only change.

## 3.3 (P0.3) — Fovea-geometry audit + exclusion — ✅ DONE

### What was built
- **`fovea_audit.py`** (repo root, standalone; modeled on `diagnose_unmatched.py`): reads each
  matched image's sidecar `.json` and classifies fovea geometry as
  `fallback | ok | missing_key | no_sidecar | unreadable`, aggregated by image / zone-row /
  patient, overall and per canonical split. Writes the confirmed-fallback image list to
  `processed_image_arrays_multiclass/fovea_fallback_images.txt`.
- **Exclusion wiring:** `zone_dataset.py` now carries `ZoneRecord.fovea_fallback` (read from the
  sidecar; sidecars missing the field default to `False`), plus a reusable
  `exclude_fovea_fallback(records)` helper. `train_convnext.py` gained
  **`--exclude-fovea-fallback`**, applied to `records` *before* the split so **all** splits
  (train/val/test) are cleaned — rationale: fovea fallback is an **input-corruption** problem
  (wrong pixels for the label), unlike `--exclude-tier1` which is a label question kept in
  val/test for comparability.
- **Verified end-to-end:** `exclude_fovea_fallback` drops exactly the 570 confirmed-fallback
  zone rows (7,720 → 7,150); flag appears in `train_convnext.py --help`; all edits compile.

### Audit results (recorded per GUIDE §3.3)

Total matched images: **772** (7,720 zone rows).

| status | images | zone rows | meaning |
|---|---:|---:|---|
| **fallback** | **57** | **570** | crosshair not detected → image-center geometry → **masks wrong**, exclude |
| ok | 249 | 2,490 | crosshair detected, geometry trustworthy |
| **missing_key** | **466** | **4,660** | sidecar predates the flag → **geometry UNVERIFIED** |

- **Confirmed fallback: 57 images / 570 zone rows across 47 patients** — now excludable via
  `--exclude-fovea-fallback` and listed in `fovea_fallback_images.txt`.
- Per current (stale 91-patient) canonical split, confirmed fallback lands as:
  **train 23 imgs / val 3 / test 9** (9 corrupted images sitting in the 14-patient test set).

### ⚠️ The bigger finding: 60% of images have UNVERIFIED geometry
**466 of 772 images (60%, 4,660 zone rows) have no `fovea_fallback` field** — their sidecars were
written by an older `pre_processing.py` before the flag existed. We therefore **cannot currently
tell** whether their zone masks are aligned. `--exclude-fovea-fallback` does **not** touch them
(only confirmed `True`), so they still flow into training/eval with unknown quality. In the current
test split, **70 of the 14-patient test images are "unknown"** on top of the 9 confirmed-bad.

**Recommended follow-up (not done here):** re-run `pre_processing.py --force` (at least on the
466 missing-key images) to regenerate sidecars *with* the flag, then re-audit. Until then, the true
fovea-corruption rate is a lower bound (≥57 images), and this is a live confound for any
`0.52`-ceiling claim. This should happen alongside 3.2 (the new 156-patient split) so the rebuilt
canonical split is defined over geometry-verified data.

---

## Files changed / added
- `eval_convnext_sweep.py` — 3→4 unpack (1 line).
- `zone_dataset.py` — `ZoneRecord.fovea_fallback`, `_read_zone_meta` returns it, `load_zone_records`
  populates it, new `exclude_fovea_fallback()` helper.
- `train_convnext.py` — `--exclude-fovea-fallback` arg + application + import.
- `fovea_audit.py` — new standalone audit script.
- `processed_image_arrays_multiclass/fovea_fallback_images.txt` — generated exclusion list (57 paths).

## Still open in Phase 0
- **3.2 (P0.1):** regenerate split over all 156 patients (`canonical_split_v2_156.json`).
- **3.4 (P0.2):** extend `run_patient_cv.py` to 5-fold + `--trainer`.
- **Follow-up:** reprocess the 466 missing-key sidecars, then re-audit, before freezing the new split.
