# AUGMENTATION POLICY

Canonical and binding across all phases. If a phase spec and this file disagree,
this file wins. Changes require a dated edit here, not a local override.

The policy is derived from the data's actual structure:

```
patient → visit(date) → eye(OD/OS) → { FP image, FA image }
                                        └→ 10 zones sharing ONE acquisition
                                        └→ zone geometry = (cx, cy, angle_deg)
```

Three facts drive every rule below:

1. **Stage B has no zone masks.** Geometric transforms are safe there.
2. **Stage C's zone masks are a function of `(cx, cy, angle_deg)`.** Any geometric
   transform of the image invalidates them unless the geometry is transformed
   identically.
3. **The 10 zones of an image come from one eye at one moment.** They must receive
   identical augmentation. The global (single-forward-pass) architecture enforces
   this automatically; the legacy crop pipeline did not.

---

## The matrix

| Transform | Stage B (pretext) | Stage C (downstream) | Rationale |
|---|---|---|---|
| CLAHE, L-channel in LAB, fixed params | deterministic, both modalities | deterministic, same params | Preprocessing, not augmentation. See §2. |
| Rotation / flip | **yes — identical transform applied to FP and its paired FA** | **no** | Stage B has no masks to desync. Stage C would invalidate zone geometry. See §3. |
| Scale / crop jitter | yes, synchronized across modalities | no | Same reason. |
| Background-token recomputation | **after** any geometric transform | after | Rotation changes which patches contain retina. |
| Photometric jitter (brightness, contrast) | **asymmetric — student input only; target input left canonical** | mild, symmetric | See §4. |
| Hue shift / channel shuffle | **never** | **never** | See §5. |
| Masking | not used (see Phase 1 §3) | n/a | The modality gap is the corruption. |
| OD/OS horizontal flip | already applied upstream, fixed | already applied upstream, fixed | Laterality standardization, **not** augmentation. See §6. |
| MixUp / CutMix | no | no | Tried in legacy (r-series), flatlined. Destroys spatial-label correspondence. |
| Multi-visit as extra samples | yes | yes, with patient-grouped splitting | See §7. |

---

## §1 Governing principle: augment Stage B aggressively, Stage C conservatively

Stage B is self-supervised. It has **no labels to corrupt**, so every additional
view is additional pretext signal — and with only ~790 paired images, it needs
them.

Stage C has labels that carry real noise. Augmentation there multiplies label
noise rather than adding information. It cannot manufacture signal about the
0/1/2 decision boundary.

**Corollary, stated plainly:** augmentation will not fix a weak positive-class
recall. If the positive class is failing because the fundus signature is weak or
the labels are ambiguous, no amount of augmentation helps — it amplifies the
problem. Do not reach for augmentation as a remedy for a class-level failure.

## §2 CLAHE

Apply as **fixed, deterministic preprocessing at data-prep time**, never as a
randomized per-epoch augmentation.

- Operate on the **L channel in LAB space**, not per-RGB-channel (which distorts
  colour relationships).
- Identical clip-limit and tile-grid for FA and FP, Stage B and Stage C.
- Suggested starting point: `clipLimit=2.0, tileGridSize=(8,8)`. Tune **once by
  eye** on ~10 images. Do not sweep it.

Randomizing contrast would turn a plausibly diagnostic property — leakage
brightness — into something the model must marginalize out. That contradicts the
reasoning behind §5.

Expected benefit is concentrated in the peripheral zones, which are visibly
dimmer than the posterior pole in Optos UWF captures.

## §3 Geometric transforms

**Stage B — permitted, with one hard requirement.**
The *same* transform must be applied to the FP and its paired FA before feature
extraction. A rotation applied to one and not the other breaks spatial
correspondence and makes the prediction target wrong. This is the single most
dangerous augmentation bug possible in this architecture.

Implementation requirement: sample the transform **once per pair**, apply to both,
and assert the transform parameters are shared. Add a unit test.

Suggested ranges: rotation ±15°, scale 0.9–1.1, horizontal flip.

**Stage C — forbidden in this phase.**
Rotation is *mathematically* possible: rotate `(cx, cy)` about the image centre
and add θ to `angle_deg`. It is nonetheless forbidden here because:

- `make_zone_mask` places the ONH at `cx + ONH_OFFSET_X` — **a fixed image-space
  offset that ignores `angle_deg` entirely.** Rotating the image would silently
  displace zone 9. (This is a pre-existing bug; see Phase 0 §5.)
- Compounding an augmentation on top of freshly rebuilt geometry adds risk to the
  one thing Phase 0 exists to make trustworthy.

Revisit only after Gate 0.B has passed and the ONH rotation bug is fixed.

## §4 Photometric jitter — asymmetric in Stage B

Jitter the **student** input (the FP branch). Leave the **target** input (the FA
branch) canonical.

Rationale: the target embedding should be a stable reference. Jittering the target
input injects noise directly into the regression target with no regularizing
benefit — it is label noise, not augmentation. Asymmetric view augmentation is
standard practice in joint-embedding SSL.

This is easy to implement backwards. Add an assertion.

## §5 Never touch hue or channel identity

Optos UWF is a **red-laser + green-laser composite**, not true RGB. The measured
blue-channel mean over retinal content is **1.1**. Channel identity encodes
imaging depth: red penetrates to choroid, green images the retinal surface.

Hue shifts and channel shuffles therefore destroy physically meaningful
information. They are forbidden in all stages.

Corollary: compute **corpus-specific per-channel mean/std** for normalisation.
ImageNet statistics are badly off-distribution for this data.

## §6 The OD/OS flip is not augmentation

`pre_processing.py` horizontally flips OS images to an OD-axis convention. This is
a **fixed, deterministic standardization step** applied once at preprocessing.

Do not add a second random horizontal flip on top of it in Stage C — that would
undo laterality standardization for half the batch and break the nasal/temporal
zone semantics. (Random flips in Stage B are acceptable because Stage B has no
zone semantics.)

## §7 Multi-visit structure

Corpus: 156 patients, 424 visits, median 2 visits/patient, max 13.

- Multiple visits of one patient are **genuine additional samples** — different
  acquisition, often different disease state — not duplicates. Use them.
- They are **not** a leakage risk provided splitting is patient-grouped.
  `zone_dataset.py::patient_split` and `run_patient_cv.py` already group on
  `patient_id`. Do not replace this logic.
- Do **not** treat visits as an augmentation multiplier when reporting effective
  dataset size. The independent unit for generalisation is the **patient** (156),
  not the visit (424) and not the zone row (7720).
