# DESCRIPTION — Full-image backbone → masked zone pooling → cross-zone attention → 10 per-zone logits

**Investigation ID:** `01_full-image-masked-pooling-cross-zone-attention`
**Corresponds to:** Option A (★ recommended) in `KANBAN/01_NEXT_STEP_OPTIONS.md`
**Status:** Committed direction — building. Not an open question; the decision is made.
**Created:** 2026-07-10
**Branch of record:** `feature/clip-convnext` (create a dedicated feature branch per experiment; see GUIDE.md)
**Reads on top of:** `KANBAN/00_SITUATION_AND_CONTEXT.md` (the honest state-of-play) and
`KANBAN/01_NEXT_STEP_OPTIONS.md` (the decision analysis that selected this option).

> If you are an AI agent picking this up cold: read the two `KANBAN/0*_*.md` docs first, then this
> file, then `GUIDE.md`. Do not re-derive the decision — it is settled. Your job is to build,
> measure honestly against the CV baseline, and report.

---

## 1. One-paragraph summary

We are replacing the current **per-zone black-masked-crop** classifier with the architecture the
project originally pitched but never actually built: a **single forward pass over the whole fundus
image**, from which we **masked-average-pool one feature token per anatomical zone**, then let those
10 zone tokens **interact through one small attention layer**, and finally emit **10 per-zone
binary logits** trained with a summed per-zone loss. The goal is to break the flat **0.519 ± 0.012
macro-F1 / ~0.40 class-1-recall** CV ceiling by fixing the actual root cause — the model currently
never sees whole-image context and never reasons jointly across zones — while keeping the per-zone
localization output that is the paper's scientific contribution.

---

## 2. Why we are doing this (the conceptual/logical case)

### 2.1 What is actually broken

The scientific target (from `00_SITUATION_AND_CONTEXT.md §1`) is **per-zone leakage severity from
fundus photographs**, graded by a clinician on FA across **10 anatomical zones** into tiers
**0 / 1 / 2**, collapsed to **binary** (`0 → 0`, `{1,2} → 1`) at the zone level. FA never enters the
network; it only produced the labels. Evaluation is **patient-level** (zones from one eye must not
leak across splits).

Two structural problems, both documented and evidence-backed, cap performance:

1. **The input deletes the signal (representation failure).** The brief promised "full image →
   feature map → masked pooling per zone," selling point: *"no information from surrounding zones is
   discarded."* **The code does the opposite.** In `zone_dataset.py::_load_zone_image` →
   `extract_zones.apply_zone_and_crop`, each zone's wedge is isolated by **setting alpha to 0
   everywhere outside the wedge**, cropping to its bounding box, and converting to RGB. The network
   receives a lone pie-slice of retina on a **black background**, resized to 288×288, classified as
   an **independent sample** with a learned `nn.Embedding(11, 64)` zone-id vector concatenated to
   the pooled features. Global vascular context is **physically deleted before the pixels reach the
   network**, and it introduces a domain shift (ImageNet backbones have never seen black-masked
   wedges).

2. **Zones are treated as i.i.d. when the disease is spatially coherent (correlation failure).**
   Uveitis leakage propagates along vasculature and shows up as a **pattern across zones**. The
   disease signal lives in the **correlation structure between zones**, not in any single zone in
   isolation. The current pipeline turns one patient into 10 independent samples; the loss, the
   sampler, and the metric all assume independence.

### 2.2 Why the 26 prior runs could not fix it

`00_SITUATION_AND_CONTEXT.md §4` documents 26 protocol iterations (r01–r26, one change per run). The
best single-split config (**r14**: ConvNeXt-Tiny, focal γ=3, inverse class weights, exclude-tier1,
20 epochs) hit test F1 **0.5599** / class-1 recall **0.5602**, but its honest **3-fold patient CV is
0.519 ± 0.012** (class-1 recall 0.402 ± 0.046). The tight fold std is the tell: **this is a flat
ceiling, not a tuning problem.** Every loss/weighting/sampling/inference trick either failed or
collapsed a class. The team's converged diagnosis (correct): *those were loss-side compensations for
a representation-side problem — the amplifier was tuned; the microphone was the problem.*

### 2.3 Why this specific architecture (and not the alternatives)

From the scorecard in `01_NEXT_STEP_OPTIONS.md`, this option is the **only** one that attacks *both*
halves of the root cause **and** keeps the per-zone contribution **and** is safe at ~156 patients:

- **Whole-image context** is preserved by construction — nothing is blacked out. (Fixes problem 1.)
- **Cross-zone attention** lets zone-*k*'s prediction depend on zones 5/7/8. (Fixes problem 2.)
- **Per-zone output is retained** — 10 logits, per-zone labels — so the localization claim stands.
- **It's the design already justified in the brief**, so a reviewer needs no new argument; the one
  honest addition (cross-zone interaction) is motivated directly by our strongest finding (spatial
  coherence).
- Rejected alternatives, briefly: **F** (context-padded crops) fixes context but not correlation;
  **B** (frozen r14 features + aggregator) models correlation over already-poisoned features, so it's
  ceiling-capped; **C** (pure patient ABMIL) deletes per-zone output and shrinks the test set to 14
  numbers; **E** (weak-seg leakage map) is highest-upside but too data-hungry for now — parked as a
  later extension on top of this. **D** (auxiliary patient head) is *not* rejected — it is the
  planned low-cost add-on **once A trains cleanly** (see §6).

### 2.4 What removing the crop buys us, concretely

- The **black-mask domain shift disappears** (nothing is masked in pixel space).
- Inference becomes **~10× cheaper** — one forward pass per eye instead of ten crops.
- The **i.i.d. assumption is gone** — zones interact, and (via the future D head) roll up to a
  patient-level prediction.
- The **interpretability story improves**: per-zone predictions **+ attention weights over zones**
  ("which zones drove this call") — a strictly richer figure set than a per-zone F1 table.

---

## 3. What we are building (the technical design)

### 3.1 Target data path

```
full fundus image (single forward pass, NOT 10 crops)
  → ConvNeXt feature map                         # global vascular context preserved
  → project the 10 zone masks onto the feature map
  → masked average pooling per zone              # 10 zone tokens from a shared context
  → 1 small attention/transformer layer over the 10 tokens   # THE NEW PART: inter-zone interaction
  → 10 per-zone binary logits, summed per-zone loss          # keeps per-zone supervision
```

### 3.2 What changes vs. the current code

The current implementation (accurate as of this writing):

- **Dataset** (`zone_dataset.py`): `ZoneImageDataset.__getitem__` returns **one masked-and-cropped
  zone image** per row, plus `label`, `zone_number`, and optional `sample_weight`. Records come from
  `load_zone_records(csv, data_root)` as `ZoneRecord(patient_id, zone_number, raw_label, label, ...)`.
  Crops are produced on the fly by `_load_zone_image` → `extract_zones.make_zone_mask` +
  `apply_zone_and_crop` from a cached cleaned `.npy` per FP image (fovea geometry in a sidecar `.json`).
- **Model** (`train_convnext.py::ZoneAwareConvNeXt`): torchvision `convnext_tiny`, global avgpool →
  flatten → LayerNorm, with the final Linear replaced by `Identity`; a learned
  `nn.Embedding(NUM_ZONES+1, zone_embed_dim)` is concatenated to the pooled 768-d features; head is
  `LayerNorm → Dropout(0.3) → Linear(→256) → GELU → Dropout(0.2) → Linear(256→2)`. `forward(images,
  zone_numbers)` takes **one image + one zone id** per sample.
- **Batching** (`batch_to_device`): returns `(images, labels, zones, sample_weight)` — note it now
  returns **4** values (added in commit `ee22b6f`); see the known bug in §7.
- **Optimizer**: AdamW, backbone at `lr * 0.1`, cosine schedule, AMP, grad-clip 1.0, early stop on
  val macro-F1.

The Option A build requires the following **new** pieces (this is the engineering surface):

1. **A new dataset item = `(full_image, 10 zone masks, 10 labels, zone geometry)` per patient-eye**,
   not 10 independent crops. Masks must be produced at the **feature-map resolution** the pooling
   layer reads from (downsample the pixel-space wedge masks to the chosen stage's grid). Reuse
   `extract_zones.make_zone_mask` for the pixel-space wedge geometry; do **not** call
   `apply_zone_and_crop` (that is the crop path we are removing).
2. **A pooling module**: given a backbone feature map `[B, C, H, W]` and 10 downsampled zone masks
   `[B, 10, H, W]`, produce 10 zone tokens `[B, 10, C]` by masked average pooling (guard against
   empty masks — see §4).
3. **A tiny interaction module**: 1 transformer/attention layer (1–2 heads) or ABMIL-style gated
   attention over the 10 tokens → 10 refined tokens.
4. **A per-zone head**: shared `Linear(C → 2)` (or `→1` with BCE) applied to each of the 10 refined
   tokens → `[B, 10, 2]` logits.
5. **A per-zone loss**: sum (or mean) of the per-zone loss over the 10 zones, **masking out zones
   that are absent/unlabeled for that eye**. Keep the current per-zone loss recipe:
   **focal γ=3 + inverse class weights** (validated; do not relitigate — `01_NEXT_STEP_OPTIONS.md`
   "Keep from what already works").
6. **New training entry point** (e.g. `train_zone_attention.py`) rather than mutating
   `train_convnext.py`, so the crop baseline stays runnable for comparison. Preserve the existing
   wandb wiring (project `uveitis-per-zone`, `--no-wandb` opt-out, per-epoch + test logging).

### 3.3 The two implementation details that decide whether it works

These are called out in `01_NEXT_STEP_OPTIONS.md` as make-or-break and are **not optional**:

1. **Feature-map resolution vs. zone size — the single biggest risk.** ConvNeXt at **stride 32**
   turns a 288px image into a **9×9 = 81-cell** grid. The **inner zones are tiny** and would get
   **<1 cell of support** — masked pooling would be degenerate (or empty) for them. The build
   **must** pool from a **higher-resolution stage (stride 8–16)** or an FPN fusion, and/or raise
   input resolution to **~512px** (→ 16×16 at stride 32). If Option A "doesn't beat 0.52," the first
   suspect is **under-resolved inner zones**, not the idea. Bake this in from the start; the brief
   hand-waved it at stride 32.

2. **Domain-informed positional encoding.** We *know* each zone's geometry (angle, radius, adjacency
   to fovea/optic-nerve-head). Feed that geometry as the positional encoding into the attention layer
   **instead of a bare learned zone index**. This lets attention learn things like "adjacent nasal-
   ring zones co-leak" — a small, novel, clinically motivated contribution rather than a generic
   transformer. The geometry constants live in `extract_zones.py`
   (`make_zone_mask`, `make_masks`, and constants like `PX_PER_MM`, ONH offset).

---

## 4. Known technical hazards specific to this build

- **Empty / degenerate zone masks at feature resolution.** After downsampling to the pooling grid,
  small inner zones may map to **zero active cells**. Masked pooling must handle a zero-sum mask
  without NaN (e.g. fall back to nearest cell, or bilinearly sample the mask and normalize by mask
  sum + ε). This is the mechanism by which detail (1) fails silently.
- **Fovea geometry fallback poisons the masks (shared with the crop pipeline).** When the yellow
  crosshair isn't detected, `pre_processing.py` falls back to **image-center** geometry
  (`fovea_fallback=True`) and keeps the image, so **every zone mask on that image is on the wrong
  pixels**. This corrupts *both* the labels and the masked pooling. The count of `fovea_fallback=True`
  sidecars has never been quantified. **This must be measured and those images excluded/re-registered
  as part of Phase 0** (see §5) — otherwise Option A re-hits ~0.55 for reasons unrelated to modeling.
- **Hard-coded geometry constants** (`PX_PER_MM=53`, ONH offset +270px in `extract_zones.py`) may not
  transfer across cameras/resolutions; they affect both mask placement and the positional encoding.
- **Zone coverage per eye is not guaranteed to be all 10.** The loss and pooling must mask out zones
  that have no row/label for a given patient-eye rather than assuming a dense 10-vector.

---

## 5. Prerequisite: Phase 0 foundations (ship *with* this, not after)

From `01_NEXT_STEP_OPTIONS.md §"Phase 0"` — these are **corrections that must land regardless**, or
any new architecture will re-hit ~0.55 for reasons unrelated to modeling. They also define the
**honest denominator** this architecture is measured against:

- **P0.1 — Regenerate the patient split over all 156 patients.** We currently train on **91** and
  leave **~65 patients / 2,410 labeled zones (~31%) idle** because the canonical split
  (`splits/canonical_split.json`, seed 13) was frozen in the baseline era and never regenerated after
  the dataset grew. Rebuild a stratified patient-level split over all 156 and **freeze the new one**
  as canonical.
- **P0.2 — Switch all comparisons to k-fold patient CV.** Single-split numbers have lied three times
  (r14 0.56 vs CV 0.52; r26 val 0.58 vs test 0.44). The **only** valid comparison is **CV-to-CV
  against 0.519**, 5-fold, stratified by patient positivity. Harness exists at `run_patient_cv.py`
  (currently 3-fold, `--no-wandb`); extend to 5-fold and wire the new model in.
- **P0.3 — Quantify and fix the fovea geometry.** Count `fovea_fallback=True` sidecars in the
  training set; exclude or re-register those images. (Also see §4 — this build depends on correct
  masks even more than the crop pipeline did.)
- **P0.4 — Fix `eval_convnext_sweep.py:116`** (unpacks 3 values from `batch_to_device`, which now
  returns 4 — see §7) so post-hoc eval works.

**These change the baseline itself.** It is entirely possible P0.1 + P0.3 alone move the CV number,
because we'd be training on ~70% more patients with cleaner geometry. That is the honest denominator
the architecture must beat — not the old 0.519 computed on the stale 91-patient split.

---

## 6. Planned follow-on: Option D auxiliary patient-level head

Once Option A trains cleanly, add **Option D** (recommended low-cost add-on): on the *same* shared
backbone and the *same* 10 zone tokens, attention-pool the 10 tokens into one patient vector and
predict a **patient-level** label (patient positive if any zone is tier-2). Train with
`per-zone loss + λ · patient loss`. It costs a few lines given A exists, acts as an explicit
correlation regularizer, yields **interpretable per-zone importance weights** without giving up
per-zone labels, and adds a **second, patient-level evaluation axis** (clinically the real question:
"does this eye need treatment?"). `λ` needs tuning. Track D as a **separate experiment sub-folder**
under this investigation once A validates.

---

## 7. Pre-existing bug to be aware of (blocks post-hoc eval)

`eval_convnext_sweep.py:116` unpacks **3** values from `batch_to_device`, but that function now
returns **4** (`images, labels, zones, sample_weight`) since commit `ee22b6f`. The threshold/TTA
sweep will crash if re-run as-is. This is P0.4 above; fix it before relying on any sweep script.

---

## 8. Expected results & success criterion (decide honestly, up front)

**Hypothesis.** Giving the model whole-image context + cross-zone interaction addresses the *actual*
root cause, so we expect it to **beat the re-baselined CV number by more than the fold std** and,
critically, to **lift class-1 recall out of the ~0.40 rut** (finding *more* than half of positive
zones). We also expect **cleaner behavior on inner zones once resolution is fixed** (detail 1) and
**interpretable attention weights** that concentrate on clinically adjacent zones.

**Success criterion (from `01_NEXT_STEP_OPTIONS.md`).** Option A is a win **only if**, on the
**new 5-fold patient CV** (Phase 0), it beats the re-baselined ConvNeXt CV number by a margin
**larger than the fold std**, **and** lifts class-1 recall clearly above ~0.40. **Single-split
improvements do not count** — they have burned us three times.

**Falsification / what a null result means.** If Option A, built with high-resolution pooling and
geometry-aware positional encoding, does **not** clear that bar, then the ceiling is **data/label
quality** (`00_SITUATION_AND_CONTEXT.md §6`: fovea fallback, 31% unused data, tiny/ambiguous eval),
**not** architecture — and the next move is **labeling/data work, not more modeling.** Record that
outcome honestly; a clean negative here is a real finding that redirects the whole project.

**What we explicitly keep (do not relitigate):** exclude-tier1 in training (biggest historical gain),
focal γ=3 + inverse class weights as the per-zone loss, and **ConvNeXt-Tiny** as the backbone (CLIP-L,
EfficientNet, RETFound all lost at this data scale; the gain here comes from *how features are pooled
and combined*, not from a bigger backbone).

---

## 9. Deliverables for this investigation

- A working `train_zone_attention.py` (or equivalent) implementing §3, with wandb wired like the
  existing trainers.
- A 5-fold CV result for **(a) the re-baselined crop ConvNeXt** and **(b) Option A**, both on the new
  156-patient split, with fold mean ± std for macro-F1 and class-1 recall.
- The fovea-fallback count and the disposition of those images.
- A short results note appended here (or in a per-experiment `RESULTS.md`) stating whether the
  success criterion was met, with the wandb run URLs.
- Attention-weight visualizations for a few eyes (the interpretability payoff).

See `GUIDE.md` in this folder for the exact commands to run.
