# Research loop log (feature/clip-convnext)

Baseline best before loop: `clip_convnext_10ep_baseline` test.macro_f1=0.5251

Canonical split: `splits/canonical_split.json` (from baseline run, seed=13).

**Current best:** `protocol_r03_convnext_focal` test.macro_f1=**0.5285**

| Run | Name | Hypothesis | test.macro_f1 | Δ vs best | Decision |
|-----|------|------------|---------------|-----------|----------|
| — | (prior best) | — | 0.5251 | — | — |
| 1 | protocol_r01_convnext_soft_ce | Soft CE + multiclass soft labels | 0.4849 | -0.0402 | Below baseline → tried CLIP |
| 2 | protocol_r02_clip_default | CLIP Large-D defaults, binary 20ep | 0.5259 | +0.0008 | Slight gain → exp 3 compare |
| 3 | protocol_r03_convnext_focal | ConvNeXt-Tiny focal vs CLIP (exp 3) | **0.5285** | +0.0034 | **NEW BEST** — ConvNeXt wins on same split |
| 4 | protocol_r04_clip_focal | CLIP + focal loss | 0.4990 | -0.0295 | No gain (val OK, test weak) |
| 5–10 | (in progress) | CLIP tune + ConvNeXt zone128 + CLIP soft labels | — | — | resuming run 5… |
| 5 | (running) protocol_r05_clip_freeze3 | CLIP unfreeze backbone after 3 epochs | — | — | — |
