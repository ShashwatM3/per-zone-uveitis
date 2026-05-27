# Research loop log (feature/clip-convnext)

Baseline best before loop: `clip_convnext_10ep_baseline` test.macro_f1=0.5251

Canonical split: `splits/canonical_split.json` (from baseline run, seed=13).

| Run | Name | Hypothesis | test.macro_f1 | Δ vs best | Decision |
|-----|------|------------|---------------|-----------|----------|
| — | (prior best) | — | 0.5251 | — | — |
| 1 | protocol_r01_convnext_soft_ce | Soft CE + multiclass soft labels | 0.4849 | -0.0402 | No improvement; val~test → try CLIP (exp 2) |
