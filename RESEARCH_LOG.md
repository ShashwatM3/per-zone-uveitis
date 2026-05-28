# Research loop log (feature/clip-convnext)

Baseline best before loop: `clip_convnext_10ep_baseline` test.macro_f1=0.5251

Canonical split: `splits/canonical_split.json` (from baseline run, seed=13).

| Run | Name | Hypothesis | test.macro_f1 | Δ vs best | Decision |
|-----|------|------------|---------------|-----------|----------|
| — | (prior best) | — | 0.5251 | — | — |
| 1 | protocol_r01_convnext_soft_ce | Soft CE + multiclass soft labels | 0.4849 | -0.0402 | No improvement; val~test → try CLIP (exp 2) |
| 2 | protocol_r02_clip_default | CLIP Large-D defaults, binary 20ep | 0.5259 | +0.0008 | IMPROVED → compare ConvNeXt (exp 3), tune CLIP |
| 3 | (pending) protocol_r03_convnext_focal | queued | — | — | — |
