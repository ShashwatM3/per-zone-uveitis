# 00 — Situation & Context: Where This Project Actually Stands

**Date:** 2026-07-10
**Branch:** `feature/clip-convnext`
**Author:** Working analysis (Claude), from full codebase + research-log + git-history + data audit review.

> Purpose of this doc: an honest, evidence-backed snapshot of what we set out to do, what we
> actually built, what we've tried, what's blocking us, and *why* — so the "next steps" doc
> (`01_NEXT_STEP_OPTIONS.md`) can be read without re-deriving any of it. No optimism tax, no
> pessimism tax. Just the state of play.

---

## 1. The goal (unchanged, still correct)

Cross-modal supervision. Fluorescein Angiography (FA) is the invasive gold standard for grading
retinal inflammation (uveitis). We want to predict its **per-zone leakage severity** labels from
**standard fundus photographs (FP)** alone, so screening can scale without FA.

- A clinician graded each of **10 anatomical zones** on FA with a severity tier: **0 (healthy),
  1 (ambiguous/borderline), 2 (active disease)**.
- Those FA-derived labels are the answer key. **FA never enters the network as pixels** — it only
  determined `Y`. At inference the model sees only fundus.
- Training was reduced to **binary** at the zone level: tier `0 → class 0`, tiers `{1,2} → class 1`.
- Evaluation is **patient-level split** (correlated zones from one eye must not leak across splits).

The scientific contribution, as written in the README and the architecture brief, is **per-zone
localization** of leakage from fundus — not just "does this patient have uveitis." That distinction
matters for every decision downstream.

---

## 2. The data reality (verified against the CSV + split files, not the summary doc)

| Fact | Value |
|------|-------|
| FP images on disk | 792 |
| FP images matched to labels | 772 (20 unmatched — filename mismatch in the Excel `UWFFP` column) |
| Zone rows in `zone_training_table.csv` (multiclass) | **7,720** |
| Unique patients in the labeled CSV | **156** |
| Patients named in `splits/canonical_split.json` | **91** (63 train / 14 val / 14 test) |
| Zone rows actually **used** by the canonical split | **5,310** |
| Zone rows **never used by any run** | **2,410 (~31%, ≈65 patients)** |
| Label balance (binary) | ~62% class 0 / ~38% class 1 |
| Tier-1 ("ambiguous") share | ~15% of zones |

**Two things the summary docs bury:**

1. **~31% of the labeled data is idle.** The canonical split was frozen early (seed 13, baseline
   era) and kept sacred for run-to-run comparability. But the dataset grew afterward (many
   `Patient300–365` were added) and the split was never regenerated. So every experiment that fought
   a "≈110-patient data ceiling" did so **with a third of the labels sitting unused**. The r14
   `metrics.json` embeds train=3046/val=750/test=840 = 4,636 rows; the *same* 91 patients now map to
   5,310 rows — i.e. the split predates the current data even for the patients it *does* include.

2. **Evaluation is tiny.** Val and test are **14 patients each**. One test patient ≈ 7% of the
   metric. This is the root of most of our false signals (see §4).

---

## 3. What we actually built (vs. what we pitched)

**This is the single most important structural fact in the project, and it is not what the
Technical Architecture Brief describes.**

- **The brief (§3.3–3.4)** proposed: *one forward pass over the full fundus image* → project the 10
  zone masks onto the ConvNeXt feature map → **masked average pooling** per zone. Its selling point:
  *"no information from surrounding zones is discarded — the backbone sees the complete fundus image
  before pooling isolates zone-specific representations."*

- **The code actually does the opposite.** In `zone_dataset.py::_load_zone_image` →
  `extract_zones.apply_zone_and_crop`: each zone's wedge mask is applied by **setting alpha to 0
  everywhere outside the wedge**, cropping to the wedge bounding box, and converting to RGB — so
  **everything outside the zone becomes black.** The model receives a lone pie-slice of retina on a
  black background, resized to 288×288, classified as an **independent sample**, with a learned
  `nn.Embedding(11, 64)` zone-id vector concatenated to the pooled features.

**Consequences of the crop-based implementation:**
- Global vascular context — the entire premise of the brief — is **physically deleted** before the
  network sees the pixels. A vessel leaking across the zone-5/zone-7 boundary is cut in half and
  blacked out.
- It induces an extra **domain shift**: an ImageNet-pretrained backbone has never seen black-masked
  wedges.
- One patient becomes 10 i.i.d. samples. The loss, the sampler, and the metric all treat zones as
  independent — which they are not.

Model head (both ConvNeXt and RETFound variants): `LayerNorm → Dropout(0.3) → Linear(→256) → GELU →
Dropout(0.2) → Linear(256→2)`, AdamW with backbone at 0.1× head LR, cosine schedule, AMP, grad-clip
1.0, early stop on val macro-F1. Zone crops are computed on the fly from a cached cleaned `.npy` per
FP image (fovea geometry stored in a sidecar `.json`).

---

## 4. What we've tried (26 protocol iterations) and what it told us

An **autonomous research agent** (see `PROTOCOLS/RESEARCH_PROTOCOL.md`) ran r01–r26, one change per
run, logged in `RESEARCH_LOG.md`. The honest summary:

| Lever pulled | Runs | Result |
|--------------|------|--------|
| **Best config** — ConvNeXt-Tiny, focal γ=3, inverse weights, **exclude tier-1**, 20ep | r14 | **test F1 0.5599, class-1 recall 0.5602** (best single-split) |
| Backbone swaps — CLIP ConvNeXt-L, EfficientNet-B3, RETFound | r02–r06, r18, r19, r26 | All ≤ ConvNeXt-Tiny. CLIP-L too big for 90 patients; EfficientNet NaN-collapsed; RETFound below ConvNeXt |
| Imbalance / loss tricks — γ sweep, effective weights, oversample, balanced batches, BBFL | r11, r15–17, r22, r23, r25 | γ=3 sweet spot; everything else either no gain or **collapsed one class** (remove weights → cls-1 recall 0.14; oversample 2× → cls-0 recall 0.14) |
| Label handling — soft labels, tier-confidence weights, exclude-tier1 | r01, r12, r17, r14 | **exclude-tier1 was the only real gain**; soft labels & tier weights failed |
| Inference tricks — threshold sweep, TTA, MixUp | r15, r16, r21 | No free lunch. Threshold stays 0.50; TTA hurt; MixUp hurt |

**The two results that actually matter:**

1. **3-fold patient CV of the r14 config = 0.519 ± 0.012** (class-1 recall 0.402 ± 0.046). The tight
   std across folds is the tell: **this is a flat ceiling, not a tuning problem.** The single-split
   0.5599 was a favorable draw — the gap to 0.519 is ~1–2 test patients flipping.

2. **r26 (RETFound, corrected recipe): val F1 0.5806 (best val ever) but test F1 0.4477, class-1
   recall 0.232.** A 0.13 val/test gap on 14-patient sets. This proved that **single-run test
   numbers are unreliable at this scale** and that better features alone didn't move the floor.

**Diagnosis the team converged on (correct):** r14→r25 was *loss-side compensation for a
representation-side problem*. We kept changing how loudly we punish class-1 errors and never changed
what the model sees. The amplifier was tuned; the microphone was the problem.

---

## 5. The core insight (from the strategy brainstorm)

Uveitis leakage is **spatially coherent** — it propagates along vasculature and appears as a pattern
*across* zones, even though the labels are recorded *per* zone. The disease signal lives in the
**correlation structure between zones**, not in any single zone in isolation.

**Refinement to that insight (important):** the conclusion is *not* "stop doing per-zone
classification." The labels have genuine per-zone variance (leakage really is localized to specific
quadrants), and per-zone output is the scientific contribution. What's broken is the **per-zone
*input* isolation** — the black-masked wedge. The correct target is **per-zone output with
whole-image input/context.** That is, coincidentally, *the architecture we originally pitched in the
brief* — plus a cross-zone interaction layer we never had.

---

## 6. Confounds that could be inflating the "0.52 ceiling"

Before attributing everything to architecture, three measured/known issues could be silently capping
performance, and **no architecture change fixes them**:

1. **Fovea geometry fallback.** When the yellow crosshair isn't detected, `pre_processing.py` falls
   back to *image-center* geometry (`fovea_fallback=True`) and keeps the image. Every zone mask on
   those images is attached to the **wrong pixels** — label says "zone 3," pixels are zone 2/4. The
   count of `fovea_fallback=True` sidecars in the training set has never been quantified. Zone
   geometry uses **hard-coded constants** (`PX_PER_MM=53`, ONH offset +270px) that may not transfer
   across cameras/resolutions.

2. **31% unused data** (see §2). We are data-starved *by our own frozen split*, not by the dataset.

3. **Tiny eval + tier-1 in the test set.** 14-patient val/test drives the variance. Tier-1
   ("ambiguous") zones are hard-labeled positive in val/test, so they contaminate the *evaluation*,
   not just training.

4. **A latent code bug:** `eval_convnext_sweep.py:116` unpacks 3 values from `batch_to_device`, which
   now returns 4 (it gained `sample_weight` in commit `ee22b6f`). The threshold/TTA sweep will crash
   if re-run as-is.

---

## 7. Where we are, in one paragraph

Honest CV-validated performance is **0.519 ± 0.012** macro-F1 with class-1 recall ~0.40 — the model
correctly finds fewer than half of positive zones, and this is stable across every loss/weighting
configuration we've tried. The parameter-sweep approach is exhausted. The diagnosis is a
**representation failure driven by input design**: we feed the model context-free black-masked zone
wedges and then ask it to detect a disease whose signal is defined by cross-zone spatial patterns.
The fix is architectural (give the model whole-image context and let zones interact), not another
loss knob. Two data/geometry confounds (unused 31%, fovea fallback) must be cleared alongside, or a
new architecture will re-hit ~0.55 for reasons unrelated to modeling. The next-step options and the
recommendation are in `01_NEXT_STEP_OPTIONS.md`.
