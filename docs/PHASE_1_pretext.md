# PHASE 1 — Cross-modal self-supervised pretext

**Status:** blocked by Gate 0.
**Gate:** `docs/ACCEPTANCE_GATES.md` §Gate 1 (sub-gates A, B)
**Ledger:** `KANBAN/PHASE_1_pretext/`
**Branch:** `x-jepa`, off `feature/binary-zone-classification`

---

## 1. Objective and framing

Train a single encoder to predict, **in latent space**, the representation of the
paired FA from the FP of the same eye — and symmetrically, the FP from the FA.
No labels. FA is used at training time only; it never appears at inference.

**This is self-supervised learning, not knowledge distillation.** The distinction
is load-bearing for this project, since another lab member is running a
distillation approach, and it rests on two design commitments:

| | Distillation | This design |
|---|---|---|
| Target source | A separate, **pretrained frozen teacher** | An **EMA of the student itself** |
| Encoders | Two distinct networks | **One shared encoder**, modality-conditioned |
| Direction | Teacher → student, one way | **Symmetric**, both directions |

There is no pretrained teacher anywhere in this pipeline. No DINO, no RETFound, no
borrowed encoder producing targets. The FA signal enters purely through the EMA
bootstrap, which is the same mechanism as BYOL / I-JEPA / V-JEPA.

This choice has a cost — EMA reintroduces representation collapse as a real
failure mode — which §6 addresses explicitly.

---

## 2. Architecture

```
                 ┌──────────────────────────────────────────────┐
   FP ──► +m_FP ─┤                                              ├─► tokens_FP
                 │   E_θ   (shared encoder, ConvNeXt-Tiny)      │
   FA ──► +m_FA ─┤                                              ├─► tokens_FA
                 └──────────────────────────────────────────────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                   P_ψ(tokens_FP) ──► ẑ_FA   P_ψ(tokens_FA) ──► ẑ_FP
                          │                     │
                          │  dense L1           │  dense L1
                          ▼                     ▼
                   sg(E_θ̄(FA, +m_FA))    sg(E_θ̄(FP, +m_FP))
                          └── EMA target, stop-grad ──┘
```

**Encoder `E_θ` — ConvNeXt-Tiny, ImageNet-21k init (`fb_in22k_ft_in1k`).**
Chosen over a ViT because: it was the empirical winner across four backbones in
the legacy sweep; its spatial inductive bias suits ~790 pretraining pairs far
better than a plain ViT; and it is definitively not a DINO checkpoint.

**Modality conditioning.** A learned per-modality embedding vector is added as a
channel-wise bias to the stem output (FiLM-style). Two vectors, `m_FP` and `m_FA`,
of dimension equal to the stem's channel count. This is what allows one set of
weights to serve both modalities.

**Token grid.** Use the **stage-3 output (stride 16)**, not the final stage
(stride 32). At 1120 px input this gives a 70×70 grid. Justification is in §4.

**Predictor `P_ψ`.** A small conv head: 2 × (3×3 conv → LayerNorm → GELU) with
width 384, then a 1×1 projection to the target channel dimension. Shared across
both directions; modality of the *target* is supplied via the same embedding
mechanism.

**Target `E_θ̄`.** Exponential moving average of `E_θ`. Momentum 0.996, cosine-annealed
to 1.0 across training. Stop-gradient on this branch. No gradients ever flow into
`E_θ̄`.

---

## 3. Objective

**No masking — this is a deliberate simplification, not a forced constraint.**

The underlying fact is well established: naive masking genuinely breaks CNNs.
Convolution cannot handle irregular random-masked input; zeroing masked regions
shifts the pixel value distribution; and the mask pattern itself vanishes after
several convolutions, a problem that is particularly acute in modern deep
convnets with many successive blocks. SparK's own ablation found that replacing
sparse masking with naive zero-outing degraded performance until it nearly
reached the plain supervised baseline — i.e. naive masking erased almost the
entire benefit of pretraining.

There are therefore two valid options:

1. **Adopt sparse convolutions** (SparK-style). This is the established fix, works
   on ConvNeXt without backbone modification, and CNN-JEPA demonstrates the exact
   architecture in this spec — shared-architecture context/target encoders, sparse
   convolutions in the context encoder, latent prediction of masked patches.
2. **Omit masking entirely.** In a cross-modal setting the modality gap is itself
   the corruption; predicting FA structure from FP is already a hard task.

**Decision: option 2.** Option 1 is more principled but is real implementation
work whose benefit cannot be measured without an ablation this project has no
budget for. Omitting masking removes the sparse-conv dependency and a whole class
of implementation risk.

Revisit option 1 if Gate 1.B fails and escalation reaches rung 3.

**Known tension, recorded rather than hidden:** SparK found loss-on-masked-patches-only
outperformed dense loss for their setting, while V-JEPA 2.1 found dense
all-position loss essential. These are different objectives (pixel reconstruction
vs latent prediction) and different downstream goals (classification vs dense
localization). Phase 2 pools per zone, which is the localized kind, so the dense
loss below follows V-JEPA 2.1. But the evidence does point both ways and this
should be stated in any writeup.

**Dense loss over all spatial positions.** For each direction:

```
L_FP→FA = (1/|V|) Σ_{i∈V} ‖ ẑ_FA,i − sg(z_FA,i) ‖₁
L_FA→FP = (1/|V|) Σ_{i∈V} ‖ ẑ_FP,i − sg(z_FP,i) ‖₁

L_pred  = ½ (L_FP→FA + L_FA→FP)
```

where `V` is the set of positions whose receptive field contains retinal content
in **both** modalities (see §5).

The loss being applied at **every** position, rather than at a sparse masked
subset, is deliberate and is the most important detail in this spec. A
sparse-target objective lets non-target positions become global aggregators and
produces feature maps with poor local structure — measured on public dense
benchmarks as roughly 22 mIoU segmentation / 0.68 RMSE depth, versus ~48 mIoU /
0.31 RMSE once all-position supervision and multi-level supervision are added.
Phase 2 pools features **per zone**, which is a dense, localized task. A
sparse-target objective would optimize for exactly the wrong property.

**Deep supervision.** Apply the same loss at stage 2 and stage 3 outputs, with
weights 0.5 and 1.0. Multi-level supervision recovers the global-recognition
quality that dense supervision alone costs.

**Anti-collapse regularization** (see §6):

```
L = L_pred + λ_var · L_variance + λ_cov · L_covariance
λ_var = 25.0, λ_cov = 1.0    (VICReg defaults; do not tune before Gate 1.A)
```

---

## 4. Resolution and zone token budget

The FP is 4000×4000 with only ~36% retinal content (median). Zone sizes at
stride 16, after cropping to the content bounding box (~2970 px):

| Zone group | 448 px | 896 px | **1120 px** | 1344 px |
|---|---:|---:|---:|---:|
| Macula (r < 159) | 7 | 29 | **45** | 65 |
| ONH (r ≈ 115) | 4 | 15 | **23** | 33 |
| Ring quadrant | 48 | 194 | **302** | 435 |
| Peripheral quadrant | 56 | 226 | **353** | 508 |

At 448 px the ONH zone pools over **4 positions** — noise, not a representation.
This is a plausible contributor to the legacy ceiling independent of the geometry
bug, since the legacy pipeline ran at 288 px.

**Decision: crop to content bbox, resize to 1120 px, stride-16 features.**

**Background pruning.** ~64% of each frame is black. Compute the retinal content
mask and exclude positions with <10% retinal coverage from both the loss and the
Phase 2 pooling. This is what makes 1120 px affordable.

Note the 84 images at 3900×3072 (patients 011–014, 016, 029, 039–040, 048,
053–055, 070). Cropping to the content bbox before resize handles the aspect
difference; assert the post-crop aspect ratio is within 10% of 1:1 and flag
otherwise.

---

## 5. Data

- Source: visit-eyes with **both** FP and FA that passed Gate 0.
  Corpus: 790 visit-eye pairs before Gate-0 exclusions.
- Split: **patient-grouped**, using the same `patient_split` logic as Phase 2.
  Pretraining must not see Phase 2's test patients. This is non-negotiable —
  pretraining on test-patient images is leakage even without labels.
- FA file: prefer `_FA_0001` (RGB, 792 files). Where only `_FA_0000` (mode `L`,
  289 files) exists, replicate to 3 channels. Record which was used per sample.
- The FA carries the yellow crosshair overlay. **Strip it before use as encoder
  input**: TIGHT mask, dilate 3 px, inpaint, and additionally **exclude those
  positions from `V`** so the loss never sees inpainted regions. Belt and braces —
  the model must not learn to predict annotation graphics.
- Registration: positions correspond across modalities only as well as Gate 0's
  transfer allows. Weight each pair's loss by its Gate-0 `od_containment_pass`
  and transfer residual, or hard-exclude pairs that failed containment.

Augmentation per `docs/AUGMENTATION_POLICY.md`: synchronized geometric transforms
applied identically to both modalities; asymmetric photometric jitter on the
student input only; no hue shifts; CLAHE deterministic.

---

## 6. Collapse: the cost of choosing SSL over distillation

A frozen teacher makes collapse impossible. An EMA target does not. This is the
one real risk introduced by the SSL framing, and it must be instrumented rather
than hoped away.

**Mitigations, all mandatory:**
- VICReg variance term: hinge loss pushing per-dimension std of the student's
  pooled embedding above 1.0.
- VICReg covariance term: off-diagonal covariance penalty decorrelating dimensions.
- EMA momentum warmup: 0.996 → 1.0 on a cosine schedule. Too-fast targets
  destabilize; too-slow targets invite collapse.

**Instrumentation, logged every epoch to W&B:**
- `embed_std_mean`, `embed_std_min`, and the fraction of dimensions with std < 0.05
- `L_variance` and `L_covariance` separately from `L_pred` — a falling total loss
  with a rising variance penalty is the collapse signature
- Deep-supervision loss at each level, separately

**Abort condition.** If >5% of embedding dimensions fall below std 0.05 for 3
consecutive epochs, the run must **fail loudly and stop**. A collapsed run that
trains to completion and reports a beautiful loss curve is the worst outcome
available here.

---

## 7. Training configuration

```yaml
image_size: 1120
feature_stride: 16
encoder: convnext_tiny.fb_in22k_ft_in1k
predictor: {depth: 2, width: 384}
ema_momentum: {start: 0.996, end: 1.0, schedule: cosine}
optimizer: AdamW
lr: 1.5e-4          # cosine, 10% warmup
weight_decay: 0.05
epochs: 300
batch_size: 8       # + grad accumulation to effective 32
grad_checkpointing: true
lambda_var: 25.0
lambda_cov: 1.0
deep_supervision: {stage2: 0.5, stage3: 1.0}
```

Hardware: 2 × RTX A6000 (48 GB) + 2 × RTX 3090 (24 GB). Run on an A6000; 1120 px
with grad checkpointing will not fit comfortably on a 3090.

**Assert and log the trainable-parameter count at run start.** A prior
`requires_grad` regression silently cut trainable parameters from ~15.6M to ~98K.
Fail the run if the count deviates >20% from expected.

---

## 8. Health checks (Gate 1.A) — run before Phase 2

**Feature coherence.** PCA of encoder spatial features, top-3 components mapped to
RGB, logged as W&B Images at ≥4 checkpoints. Coherent structure (vessels, disc,
macula separable) means the dense loss is working. Speckle means it is not, and
Phase 2 is a waste of GPU time.

**Collapse.** Per §6. Hard abort.

**Representation gate (1.B).** Frozen linear probe on zone labels using the
corrected geometry and the Phase 2 CV harness, compared against the same probe on
an ImageNet-21k-initialised encoder with no pretext training. Required margin:
**≥0.02 macro-F1, CV mean**.

If 1.B fails, escalate to rung 3 in `ACCEPTANCE_GATES.md`. Do not proceed to
Phase 2 hoping fine-tuning rescues it.

---

## 9. Honest bounds

Three constraints on how much this can deliver, stated before the run so they
are not rediscovered as excuses afterwards:

1. **~790 pairs is thin for SSL pretraining.** The dense objective helps —
   roughly 2,500 supervised positions per image rather than one — but the
   independent unit is still the patient, of which there are 156.
2. **Registration bounds the objective.** Positions correspond only as well as
   Gate 0's transfer achieved. An 86 px median offset against a 159 px inner-zone
   radius means the inner zones are the least reliable part of the signal.
3. **The modality gap may be genuine.** Leakage is dye extravasation and is
   FA-specific by construction. This objective extracts whatever fundus-visible
   correlate exists; it cannot manufacture one. Related work detects non-perfusion
   from colour fundus with reasonable accuracy but notably lower stability than
   FA-based models — suggesting the correlate is real but lossy.

None of these argue against running Phase 1. They argue that the expected gain is
bounded, and that Gate 1.B exists to detect the failure early and cheaply.
