# 01 — Next Step: Options, Evaluation, and Recommendation

**Date:** 2026-07-10
**Reads on top of:** `00_SITUATION_AND_CONTEXT.md`
**Nature of this doc:** a committed decision analysis, not a list of experiments to "go find out." We
already have the facts (26 runs, CV = 0.519 ± 0.012, the black-mask input problem, the spatial-
coherence insight, the 31%-unused-data and fovea confounds). This picks *what to build next* and
why, and rules the alternatives in or out on the merits.

---

## The decision, stated precisely

> Given per-zone labels, a spatially-coherent disease signal, ~156 labeled patients, and a 14-patient
> test set, **what architecture do we commit to next to break the 0.52 macro-F1 / 0.40 class-1-recall
> ceiling — while preserving the per-zone-localization contribution?**

Everything below is judged on five axes:

1. **Addresses the root cause** — does it give the model (a) whole-image context and (b) the ability
   to exploit inter-zone correlation? (These are the two things §5 of the situation doc identified.)
2. **Faithful to the contribution** — does it keep *per-zone* output? (Our paper claims localization.)
3. **Survives the data scale** — ~90–156 patients / ~156 "bags." Anything high-capacity overfits.
4. **Cost & risk** — engineering effort and the chance it silently fails to train.
5. **Scientific / interpretability payoff** — what it buys us in the writeup.

---

## Phase 0 — Non-negotiable foundations (ship *with* whatever we choose)

These are not "the next step" and not experiments to decide anything — they are corrections that
must land regardless, or any new architecture will re-hit ~0.55 for reasons unrelated to modeling.
They are cheap and they de-confound every future comparison.

- **P0.1 — Regenerate the patient split over all 156 patients.** We are currently training on 91 and
  leaving ~65 patients / 2,410 labeled zones idle. Rebuild the split (stratified, patient-level) and
  **freeze the *new* one** as the canonical going forward.
- **P0.2 — Switch all comparisons to k-fold patient CV.** Single-split numbers have lied to us three
  times (r14 0.56 vs CV 0.52; r26 val 0.58 vs test 0.44). The **only** valid comparison from now on
  is CV-to-CV against the **0.519** baseline. 5-fold, stratified by patient positivity.
- **P0.3 — Quantify and fix the fovea geometry.** Count `fovea_fallback=True` sidecars in the
  training set. Any zone mask built on image-center geometry attaches labels to the wrong pixels and
  poisons both training and the masked-pooling of the new architecture (which *also* depends on
  correct zone masks). Exclude or re-register those images.
- **P0.4 — Fix `eval_convnext_sweep.py:116`** (3-vs-4 unpack of `batch_to_device`) so post-hoc eval
  works. Recover the 20 unmatched FPs only if it's a trivial Excel fix (low priority).

**These change the baseline itself.** It is entirely possible P0.1+P0.3 alone move the CV number,
because we'd be training on ~70% more patients with cleaner geometry. That is not a distraction from
the architecture work — it is the honest denominator the architecture must be measured against.

---

## The options

### Option A — Full-image backbone → masked zone pooling → cross-zone attention → 10 per-zone logits  ★ recommended

**What it is.** Finish the architecture we *originally pitched*, plus the interaction layer it lacked:

```
full fundus image (single forward pass)
  → ConvNeXt feature map                      # global vascular context preserved
  → masked average pooling per zone           # 10 zone tokens, from shared context (brief §3.4)
  → 1 small attention/transformer layer over the 10 tokens   # inter-zone interaction (the new part)
  → 10 per-zone logits, summed per-zone loss   # brief §3.6, keeps per-zone supervision
```

- **Root cause:** ✅✅ both halves. Whole-image context is preserved *by construction* (nothing is
  blacked out); cross-zone attention lets zone-k's prediction depend on zones 5/7/8. This is exactly
  the "signal lives in the correlation structure" fix.
- **Faithful to contribution:** ✅ per-zone output, per-zone labels — unchanged claim.
- **Data scale:** ✅ if the interaction module is kept *tiny* (1 layer, 1–2 heads, or ABMIL-style
  gated attention). Backbone is shared across zones so it sees 10× more supervised signal per forward
  pass; the new params are few.
- **Cost/risk:** Medium. Requires restructuring the dataset to yield `(full_image, 10 zone masks)`
  per patient instead of 10 independent crops, and pooling on the feature map. No exotic training.
- **Payoff:** High + defensible. It's the design in the brief, so it needs no new justification to a
  reviewer; the attention weights are interpretable ("which zones drove this call"); it's ~10× cheaper
  at inference (one forward pass per eye, not ten crops); and it removes the black-mask domain shift.

**Two implementation details that decide whether it works:**

- **Feature-map resolution vs. zone size.** At ConvNeXt stride 32, a 288px image → 9×9 = 81 cells.
  The **inner zones are tiny** and would get <1 cell of support — masked pooling would be degenerate
  for them. Fix: pool from an **earlier/higher-resolution stage** (stride 8/16) or an FPN fusion, or
  raise input to ~512px (→16×16). This is the make-or-break knob; the brief hand-waved it at stride 32.
- **Domain-informed positional encoding.** We *know* each zone's geometry (angle, radius, adjacency
  to fovea/ONH). Feed that as the positional encoding into the attention instead of a bare learned
  index. Now attention can learn "adjacent nasal-ring zones co-leak" — a small, novel, clinically
  motivated contribution rather than a generic transformer.

---

### Option D — Option A + auxiliary patient-level head (multi-task)  ★ recommended as a low-cost add-on to A

**What it is.** On the same shared backbone and the same 10 zone tokens, add a second head:
attention-pool the 10 tokens into one patient vector and predict a **patient-level** label (patient
positive if any zone is tier-2). Train with `per-zone loss + λ · patient loss`.

- **Root cause:** ✅✅ (inherits A) + the patient head is an explicit correlation regularizer.
- **Faithful to contribution:** ✅ per-zone head is primary; patient head is auxiliary.
- **Data scale:** ✅ marginal params.
- **Cost/risk:** Low *once A exists* (a few lines). Risk: needs `λ` tuning.
- **Payoff:** The attention pooling yields **interpretable per-zone importance weights** (the good
  part of ABMIL) *without* giving up per-zone labels, and it gives us a **second, patient-level
  evaluation axis** — clinically the real question ("does this eye need treatment?"). Best of both
  framings from the brainstorm, none of the cost.

---

### Option F — Context-padded per-zone crops (minimal input fix, keep the current per-zone pipeline)

**What it is.** Stop zeroing alpha outside the zone. Instead crop a **padded bounding box** around
the zone with the surrounding retina left visible, keep the zone-id embedding to mark the target.
Everything else (independent samples, per-zone loss) stays.

- **Root cause:** ✅ context / ✗ correlation. Restores local context to the *input* but zones are
  still i.i.d.; no explicit cross-zone interaction.
- **Faithful to contribution:** ✅.
- **Data scale:** ✅ (no new params).
- **Cost/risk:** **Lowest** — a one-function change in `apply_zone_and_crop` + `_load_zone_image`.
- **Payoff:** Partial. Strictly better than the black wedge, but it doesn't model the correlation
  structure — it only lets each zone *see* its neighborhood, not *reason jointly* over zones.

**Verdict:** valuable, but as a **first increment of A, not a destination.** A full-image backbone
gives every zone context *and* enables interaction; F gives context only. Since A subsumes F, F is
worth doing only if we want a 1-day intermediate before the full rebuild.

---

### Option B — Two-stage: freeze existing r14 features → train a cross-zone aggregator on top

**What it is.** Extract the 256-d zone vectors from the frozen r14 model for all 10 zones/patient,
then train an ABMIL / small transformer over those 10 vectors.

- **Root cause:** ✗ context / ⚠ correlation. The aggregator can model correlation, **but only over
  features that were themselves computed from black-masked wedges.** It can recombine context-poor
  vectors; it cannot recover deleted context. Its ceiling is capped by the poisoned features.
- **Cost/risk:** Lowest of the "real" options (hours). But —
- **Payoff:** Structurally limited. This is the seductive-but-shallow path: cheap, and it *looks*
  like it addresses correlation, but it leaves the actual root cause (input deletion) untouched.

**Verdict:** Rejected as the primary step. Fine only as a throwaway sanity check that "aggregation
helps at all," which we don't need — we already know the features are the problem, not the head.

---

### Option C — Pure patient-level ABMIL (collapse to one label per patient)

**What it is.** Drop zone labels entirely; one label per patient; attention over 10 zones.

- **Root cause:** ✅ correlation / ✅ context (if built on full image) — but —
- **Faithful to contribution:** ❌ **This throws away per-zone localization**, which is the paper.
- **Data scale / eval:** ❌❌ It makes the test set **14 numbers**. The brainstorm sold this as
  "cleaner evaluation (23 predictions vs 1,160)"; in reality our test split is 14 patients, so it
  *amplifies* the variance that already burned us (r26). Fewer, noisier evaluation points is not
  cleaner — it's weaker statistical power on top of a smaller claim.

**Verdict:** Rejected. If we want ABMIL's benefits, Option D delivers them as an auxiliary head
without surrendering the contribution or the evaluation power.

---

### Option E — Weakly-supervised leakage segmentation → pool the predicted map over zone masks

**What it is.** Full image → encoder–decoder (FPN/U-Net) → a coarse **leakage-severity heatmap** →
pool the heatmap over each zone mask → per-zone prediction. Supervised only by the coarse per-zone
labels (no pixel masks) — i.e., weakly-supervised / CAM-style.

- **Root cause:** ✅✅ context and coherence emerge naturally from a spatial decoder.
- **Faithful to contribution:** ✅ and then some — produces a **pixel-level leakage map**, the most
  clinically compelling output we could show.
- **Data scale:** ❌ **Highest risk.** Weakly-supervised segmentation from ~156 patients of coarse
  zone labels is finicky and data-hungry; easy to get a decoder that never localizes.
- **Cost/risk:** High.

**Verdict:** The highest-upside, highest-risk option, and the best *paper* if it works. Park it as a
**research extension after A is landed**, not the next step. A working A gives us the zone tokens and
the data pipeline that E would build on anyway.

---

## Scorecard

| Option | Context | Correlation | Per-zone kept | Data-scale safe | Cost | Upside | Call |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| **A** full-image + pool + attention | ✅✅ | ✅✅ | ✅ | ✅ | Med | High | **Build** |
| **D** A + patient aux head | ✅✅ | ✅✅ | ✅ | ✅ | Low* | High | **Add to A** |
| F context-padded crops | ✅ | ✗ | ✅ | ✅ | Low | Med | Optional first increment |
| B frozen features + aggregator | ✗ | ⚠ | ✅ | ✅ | Low | Low | Reject (ceiling-capped) |
| C pure patient ABMIL | ✅ | ✅ | ❌ | ❌ | Low | Low | Reject (kills contribution + eval) |
| E weak-seg leakage map | ✅✅ | ✅✅ | ✅ | ❌ | High | Highest | Later extension |

\* Low *given A exists.*

---

## Recommendation

**Build Option A (full-image backbone → masked zone pooling → tiny cross-zone attention → 10 per-zone
logits), with Option D's auxiliary patient-level head, on top of the Phase-0 foundations.**

### Why this and not the others

1. **It is the only option that attacks *both* halves of the root cause while keeping the
   contribution.** F fixes context but not correlation; B fixes correlation over poisoned features; C
   fixes both but deletes the per-zone claim and guts the eval; E fixes both but is too risky at this
   data scale. A is the sole point that is complete *and* safe *and* faithful.
2. **It's the design we already justified.** It's the brief's own architecture — reviewers need no
   new argument for it — with exactly one honest addition (cross-zone interaction) motivated directly
   by our strongest finding (spatial coherence). We're not gambling on novelty; we're finishing the
   intended design and fixing the one thing that made it fall back to independent crops.
3. **It removes three problems at once:** the black-mask domain shift (gone — nothing is masked in
   pixel space), the 10×-redundant inference cost (gone — one forward pass per eye), and the i.i.d.
   assumption (gone — zones interact and, via D, roll up to a patient prediction).
4. **The interpretability story writes itself:** per-zone predictions + attention weights over zones
   + (via D) a patient-level call. That's a strictly richer figure set than "per-zone F1 table."

### The one thing that makes or breaks it

**Feature-map resolution vs. zone size.** Inner zones are small; masked pooling at stride 32 is
degenerate for them. The build must pool from a **higher-resolution feature stage (stride 8–16) or an
FPN**, and/or run at ~512px. If A "doesn't beat 0.52," the first suspect is under-resolved inner
zones, not the idea. Bake this in from the start.

### Keep from what already works (don't relitigate)

- **exclude-tier1** in training (validated, biggest historical gain) — keep.
- **focal γ=3 + inverse class weights** per zone — keep as the per-zone loss.
- **ConvNeXt-Tiny** backbone — keep; CLIP-L/RETFound/EfficientNet all lost at this data scale, and A's
  gain comes from *how features are pooled and combined*, not from a bigger backbone (LAW 1).

### Success criterion (decide honestly, up front)

A is a win only if, on the **new 5-fold CV** (Phase 0), it beats the re-baselined ConvNeXt number by a
margin larger than the fold std, **and** lifts class-1 recall out of the ~0.40 rut. Single-split
improvements do not count. If A clears that bar, D and then E become the paper's depth; if it doesn't,
the ceiling is data/label quality (§6 of the situation doc), and the next move is labeling/data, not
modeling.

### Sequencing

1. **Phase 0** (foundations): new 156-patient split, 5-fold CV harness, fovea audit/fix, re-baseline
   ConvNeXt under CV. → establishes the true denominator.
2. **Option A** on that foundation, pooling from a high-res stage, tiny attention, geometry-based
   positional encoding.
3. **Option D** auxiliary patient head once A trains cleanly.
4. **Option E** as a later research extension if A/D validate the direction.

*(Optional: Option F as a 1-day intermediate before A if a quick partial signal is wanted — but it is
a stepping stone, not the destination.)*
