## Section 0 — Verdict

- The yellow zone crosshair lives on the **FA** (`_FA_0001` RGB), not the FP: FA LOOSE/TIGHT detection yields balanced H/V counts (median ~7/~6, 58/60 balanced single-digit); FP LOOSE yields noise (median **2315** H / **79.5** V lines).
- `make_yellow_mask` (LOOSE) is **not selective on FP**: median **63.0%** of FP content pixels match; on FA the same mask is **0.36%** of content.
- Of **715** non-fallback stored FP `(cx,cy)` values, **390 (54.5%)** fail anatomical checks (outside content / near edge / bright tissue). Median distance from image center is **1430 px**.
- FA TIGHT succeeds on **768/789 (97.3%)** `_FA_0001` visit-eyes. Where both FP-accepted and FA-success exist (n=692), FP↔FA fovea distance median is **1442 px** (min 91, max 2426) — FP detections are systematically wrong.
- Training-row blast radius (Section 3 criteria on accepted FP detections): **3900/7720 (50.5%)** rows; including fallbacks **4470 (57.9%)**.
- Visual QC of 25 random non-fallback visit-eyes: **0/25** stored red FP foveae sit at the anatomical fovea; green FA TIGHT marks the visible FA crosshair in all cases where FA detection succeeded (24/25).

## Section 1 — Mask coverage

Sample: **60** random visit-eyes with both FP and FA (seed=2026). OS images flipped horizontally to match preprocessing. FA preferred `_FA_0001`.

Definitions:
- **LOOSE** = `extract_zones.make_yellow_mask`: `(R>40)&(G>40)&(B<100)&(|R-G|<85)&((R-B)>20)&((G-B)>20)`
- **TIGHT** = `(R>110)&(G>110)&(B<90)&((R-B)>60)&((G-B)>60)`
- Content = `mean(RGB)>4`

### Medians

| modality | content px | LOOSE count | LOOSE % content | TIGHT count | TIGHT % content |
|---|---:|---:|---:|---:|---:|
| FP | 5694537 | 3575974 | 63.00 | 101067 | 1.82 |
| FA | 5052319 | 19034 | 0.36 | 15759 | 0.30 |

**LOOSE is not selective on FP.** It matches a majority of Optos FP retinal pixels (median ~63% of content) because Optos pseudocolor has near-zero blue. On FA it matches <0.4% of content.

### Per-image table (60)

| visit | FP content | FP LOOSE | FP LOOSE% | FP TIGHT | FP TIGHT% | FA content | FA LOOSE | FA LOOSE% | FA TIGHT | FA TIGHT% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Patient11/20210615/OD | 11399324 | 4053082 | 35.56 | 5732 | 0.05 | 7471026 | 19268 | 0.26 | 15885 | 0.21 |
| Patient14/20210119/OS | 11500301 | 5151236 | 44.79 | 32681 | 0.28 | 7718416 | 19428 | 0.25 | 15876 | 0.21 |
| Patient16/20250401/OS | 5729341 | 3818307 | 66.64 | 373857 | 6.53 | 5577884 | 19121 | 0.34 | 16454 | 0.29 |
| Patient30/20250422/OS | 5716823 | 4362081 | 76.30 | 658500 | 11.52 | 5577477 | 18913 | 0.34 | 16034 | 0.29 |
| Patient300/20260106/OD | 5753618 | 4174217 | 72.55 | 57912 | 1.01 | 5600492 | 19119 | 0.34 | 15975 | 0.29 |
| Patient301/20260224/OS | 5762042 | 3266499 | 56.69 | 20808 | 0.36 | 4686962 | 19209 | 0.41 | 15693 | 0.33 |
| Patient303/20251223/OD | 5626470 | 4187617 | 74.43 | 46068 | 0.82 | 5544473 | 19034 | 0.34 | 15765 | 0.28 |
| Patient304/20250121/OD | 5889722 | 4640764 | 78.79 | 299162 | 5.08 | 5255026 | 19179 | 0.36 | 16311 | 0.31 |
| Patient310/20240305/OS | 5595366 | 4353265 | 77.80 | 281850 | 5.04 | 4766744 | 19044 | 0.40 | 15727 | 0.33 |
| Patient311/20260310/OS | 5722314 | 4263654 | 74.51 | 40064 | 0.70 | 5326792 | 19325 | 0.36 | 15844 | 0.30 |
| Patient316/20230214/OS | 5718377 | 3987272 | 69.73 | 855713 | 14.96 | 5608213 | 19295 | 0.34 | 16038 | 0.29 |
| Patient32/20220329/OS | 5732087 | 4075971 | 71.11 | 1054462 | 18.40 | 4634367 | 19035 | 0.41 | 15667 | 0.34 |
| Patient321/20251014/OD | 5752612 | 4279031 | 74.38 | 1498754 | 26.05 | 5645862 | 19142 | 0.34 | 16452 | 0.29 |
| Patient326/20201117/OD | 5594916 | 2051001 | 36.66 | 29027 | 0.52 | 4467958 | 19252 | 0.43 | 16109 | 0.36 |
| Patient328/20220412/OS | 5687704 | 4084269 | 71.81 | 34853 | 0.61 | 4763161 | 19144 | 0.40 | 15718 | 0.33 |
| Patient329/20200901/OS | 5554396 | 2831502 | 50.98 | 94355 | 1.70 | 4446444 | 18228 | 0.41 | 15084 | 0.34 |
| Patient329/20250506/OD | 5727215 | 3067962 | 53.57 | 181810 | 3.17 | 5367814 | 18972 | 0.35 | 15925 | 0.30 |
| Patient330/20200512/OD | 5460303 | 3459695 | 63.36 | 269254 | 4.93 | 4278349 | 19224 | 0.45 | 16104 | 0.38 |
| Patient337/20220712/OD | 5520470 | 2134767 | 38.67 | 39536 | 0.72 | 4940396 | 19098 | 0.39 | 15962 | 0.32 |
| Patient339/20250218/OD | 5689366 | 722394 | 12.70 | 969 | 0.02 | 5494487 | 19239 | 0.35 | 16274 | 0.30 |
| Patient344/20211005/OD | 5616177 | 3619929 | 64.46 | 88886 | 1.58 | 4343976 | 19153 | 0.44 | 15546 | 0.36 |
| Patient348/20251028/OD | 5633319 | 3198475 | 56.78 | 62624 | 1.11 | 5220890 | 19168 | 0.37 | 16035 | 0.31 |
| Patient349/20260326/OS | 5710381 | 2727654 | 47.77 | 151454 | 2.65 | 5622383 | 19348 | 0.34 | 16097 | 0.29 |
| Patient35/20210223/OS | 5696463 | 3241595 | 56.91 | 435106 | 7.64 | 4472895 | 19237 | 0.43 | 16156 | 0.36 |
| Patient351/20221108/OD | 5643563 | 3784261 | 67.05 | 7859 | 0.14 | 5037341 | 18959 | 0.38 | 16207 | 0.32 |
| Patient352/20220607/OD | 5635618 | 3463479 | 61.46 | 22463 | 0.40 | 5011500 | 19412 | 0.39 | 15826 | 0.32 |
| Patient357/20230214/OS | 5724403 | 3620372 | 63.24 | 70744 | 1.24 | 5022161 | 18943 | 0.38 | 15832 | 0.32 |
| Patient36/20220125/OD | 5559458 | 4004362 | 72.03 | 53960 | 0.97 | 5152297 | 18992 | 0.37 | 15514 | 0.30 |
| Patient360/20250422/OD | 5752260 | 4083347 | 70.99 | 55042 | 0.96 | 5628659 | 19035 | 0.34 | 15984 | 0.28 |
| Patient364/20230530/OD | 5704082 | 3510270 | 61.54 | 82538 | 1.45 | 5005721 | 19400 | 0.39 | 15753 | 0.31 |
| Patient37/20210831/OS | 5617556 | 4682146 | 83.35 | 2266779 | 40.35 | 4998562 | 16014 | 0.32 | 14049 | 0.28 |
| Patient37/20240326/OD | 5519917 | 4802506 | 87.00 | 2285799 | 41.41 | 5359336 | 15310 | 0.29 | 13560 | 0.25 |
| Patient38/20220215/OD | 5535588 | 4194259 | 75.77 | 2555148 | 46.16 | 4857868 | 16040 | 0.33 | 13730 | 0.28 |
| Patient40/20220315/OS | 5705932 | 4121240 | 72.23 | 912368 | 15.99 | 4786689 | 18736 | 0.39 | 16214 | 0.34 |
| Patient55/20210316/OS | 5554766 | 3039229 | 54.71 | 77963 | 1.40 | 4667525 | 18934 | 0.41 | 15640 | 0.34 |
| Patient56/20210309/OD | 5562461 | 2606350 | 46.86 | 5684 | 0.10 | 5067297 | 18977 | 0.37 | 15177 | 0.30 |
| Patient57/20230809/OD | 5695781 | 2901048 | 50.93 | 41062 | 0.72 | 5637677 | 16089 | 0.29 | 13688 | 0.24 |
| Patient58/20230627/OS | 5693293 | 4004206 | 70.33 | 153205 | 2.69 | 5095419 | 16524 | 0.32 | 13976 | 0.27 |
| Patient63/20231128/OS | 5727121 | 4029935 | 70.37 | 154491 | 2.70 | 5086966 | 16151 | 0.32 | 13802 | 0.27 |
| Patient65/20100922/OS | 5537571 | 2866706 | 51.77 | 121839 | 2.20 | 4887798 | 16187 | 0.33 | 13807 | 0.28 |
| Patient69/20230808/OD | 5574748 | 2732836 | 49.02 | 337438 | 6.05 | 5237855 | 19124 | 0.37 | 16529 | 0.32 |
| Patient70/20210202/OD | 5737259 | 2524197 | 44.00 | 678724 | 11.83 | 5168131 | 15460 | 0.30 | 13038 | 0.25 |
| Patient70/20230815/OD | 5757152 | 2668586 | 46.35 | 193393 | 3.36 | 4396170 | 16120 | 0.37 | 13700 | 0.31 |
| Patient76/20240409/OS | 5752684 | 2635626 | 45.82 | 160355 | 2.79 | 5249014 | 15632 | 0.30 | 13569 | 0.26 |
| Patient77/20240604/OD | 5734285 | 3138207 | 54.73 | 23570 | 0.41 | 5553903 | 16146 | 0.29 | 13793 | 0.25 |
| Patient80/20230919/OS | 5740502 | 4034251 | 70.28 | 129816 | 2.26 | 5583414 | 16319 | 0.29 | 13865 | 0.25 |
| Patient80/20240723/OD | 5554510 | 3884845 | 69.94 | 107779 | 1.94 | 4022655 | 15970 | 0.40 | 13589 | 0.34 |
| Patient80/20251021/OD | 5265003 | 2278201 | 43.27 | 36151 | 0.69 | 3661425 | 17560 | 0.48 | 11988 | 0.33 |
| Patient81/20250819/OD | 5737380 | 3696060 | 64.42 | 41458 | 0.72 | 5610983 | 15918 | 0.28 | 13651 | 0.24 |
| Patient83/20221229/OS | 5628963 | 3532018 | 62.75 | 32156 | 0.57 | 4564834 | 19207 | 0.42 | 16122 | 0.35 |
| Patient85/20251125/OD | 5736800 | 4677047 | 81.53 | 1157568 | 20.18 | 5228594 | 17758 | 0.34 | 12035 | 0.23 |
| Patient86/20210406/OD | 5455896 | 3120547 | 57.20 | 82315 | 1.51 | 4688388 | 15995 | 0.34 | 13441 | 0.29 |
| Patient86/20250819/OS | 5444485 | 3518375 | 64.62 | 91640 | 1.68 | 5234387 | 16130 | 0.31 | 13861 | 0.26 |
| Patient87/20230606/OD | 5660891 | 3770868 | 66.61 | 5461 | 0.10 | 4711572 | 16327 | 0.35 | 13865 | 0.29 |
| Patient87/20250722/OS | 5640967 | 2052600 | 36.39 | 470709 | 8.34 | 4855791 | 19227 | 0.40 | 15891 | 0.33 |
| Patient88/20250107/OD | 5523562 | 3093976 | 56.01 | 124841 | 2.26 | 4783435 | 19153 | 0.40 | 16192 | 0.34 |
| Patient92/20251104/OS | 5488162 | 2961289 | 53.96 | 132820 | 2.42 | 4585074 | 17795 | 0.39 | 12083 | 0.26 |
| Patient94/20250204/OS | 5751810 | 4004930 | 69.63 | 27834 | 0.48 | 5668726 | 19300 | 0.34 | 16375 | 0.29 |
| Patient95/20250401/OD | 5758861 | 2688126 | 46.68 | 546592 | 9.49 | 5027955 | 19324 | 0.38 | 16472 | 0.33 |
| Patient95/20250812/OD | 5742066 | 1396076 | 24.31 | 333014 | 5.80 | 4715536 | 19510 | 0.41 | 16385 | 0.35 |

## Section 2 — Crosshair presence per modality

Same 60 pairs. Detection = codebase logic (dilate 1×3×3, `HoughLinesP` rho=1 θ=π/180 thr=30 minLen=max(80,0.08·min(H,W)) maxGap=35, ±25° H/V bins) with **float** line intersection.

### Success and H/V structure

| modality × mask | success | median lines | median H | median V | median H/V | H>100 | V>100 | balanced H,V ∈[1,15] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP LOOSE | 56/60 | 2700.0 | 2315.0 | 79.5 | 29.024 | 58 | 27 | 0 |
| FP TIGHT | 26/60 | 126.0 | 0.0 | 16.0 | 0.077 | 18 | 13 | 3 |
| FA LOOSE | 58/60 | 13.5 | 7.0 | 6.0 | 1.000 | 0 | 0 | 58 |
| FA TIGHT | 58/60 | 14.0 | 7.0 | 6.5 | 1.000 | 0 | 0 | 58 |

Error tallies:

- FP LOOSE: `{'oob(-112.2,1874.3)': 1, 'oob(3991.1,1781.9)': 1, 'OK': 56, 'oob(4039.0,2746.0)': 1, 'cannot_separate_HV': 1}`
- FP TIGHT: `{'no_crosshair_lines': 5, 'cannot_separate_HV': 29, 'OK': 26}`
- FA LOOSE: `{'OK': 58, 'cannot_separate_HV': 2}`
- FA TIGHT: `{'OK': 58, 'cannot_separate_HV': 2}`

**Which modality carries the crosshair:** the **FA**. Real crosshair signature (balanced single-digit H and V) appears on FA for both LOOSE and TIGHT (58/60). FP LOOSE produces hundreds–thousands of mostly-horizontal noise lines (0/60 balanced); FP TIGHT rarely forms a usable crosshair (26/60 succeed, only 3 balanced).

### Per-image detection summary

| visit | FP LOOSE H/V/err/(cx,cy) | FP TIGHT H/V/err | FA LOOSE H/V/err/(cx,cy) | FA TIGHT H/V/err/(cx,cy) |
|---|---|---|---|---|
| Patient11/20210615/OD | 2617/44/oob(-112.2,1874.3) | 0/0/no_crosshair_lines | 9/9/OK/(1894,1621,∠-8.0) | 9/8/OK/(1893,1620,∠-8.0) |
| Patient14/20210119/OS | 3072/243/oob(3991.1,1781.9) | 0/46/cannot_separate_HV | 9/9/OK/(1949,1552,∠-3.0) | 7/8/OK/(1950,1551,∠-3.0) |
| Patient16/20250401/OS | 824/1408/OK/(3356,1681,∠9.0) | 0/346/cannot_separate_HV | 7/6/OK/(2044,1964,∠2.0) | 8/6/OK/(2044,1964,∠2.0) |
| Patient30/20250422/OS | 2494/4/OK/(461,2577,∠-14.0) | 733/33/OK/(74,2850,∠-22.0) | 6/4/OK/(2066,1867,∠3.0) | 5/3/OK/(2064,1869,∠3.0) |
| Patient300/20260106/OD | 3232/9/OK/(3589,2035,∠-11.0) | 0/14/cannot_separate_HV | 15/12/OK/(1950,2181,∠-10.0) | 13/12/OK/(1953,2175,∠-11.0) |
| Patient301/20260224/OS | 1980/153/OK/(3513,1920,∠0.0) | 0/7/cannot_separate_HV | 8/7/OK/(2062,1960,∠-6.0) | 7/7/OK/(2062,1960,∠-6.0) |
| Patient303/20251223/OD | 2049/37/OK/(3661,2499,∠7.0) | 7/70/OK/(3618,2314,∠20.0) | 4/4/OK/(1987,1963,∠-3.0) | 5/5/OK/(1990,1964,∠-3.0) |
| Patient304/20250121/OD | 9/235/OK/(1216,2637,∠17.0) | 503/32/OK/(3683,1576,∠-16.9) | 10/10/OK/(1937,1891,∠-14.0) | 9/10/OK/(1936,1892,∠-14.0) |
| Patient310/20240305/OS | 2321/32/OK/(651,1776,∠9.0) | 333/11/OK/(535,2354,∠2.0) | 8/8/OK/(2001,2011,∠-12.0) | 6/8/OK/(2001,2012,∠-12.0) |
| Patient311/20260310/OS | 1242/150/OK/(3243,1808,∠-16.0) | 0/9/cannot_separate_HV | 11/9/OK/(2032,1975,∠-25.0) | 9/13/OK/(2035,1976,∠-24.9) |
| Patient316/20230214/OS | 402/575/OK/(885,1362,∠-21.0) | 273/381/OK/(1061,899,∠13.0) | 9/10/OK/(2080,2007,∠-13.1) | 11/9/OK/(2079,2009,∠-13.0) |
| Patient32/20220329/OS | 2070/158/OK/(759,2404,∠-15.0) | 1797/20/OK/(1022,2492,∠-13.0) | 6/6/OK/(2007,1958,∠-14.0) | 7/7/OK/(2007,1957,∠-14.0) |
| Patient321/20251014/OD | 2977/18/OK/(3566,2090,∠0.0) | 1901/14/OK/(918,1340,∠14.0) | 5/5/OK/(1971,1938,∠5.0) | 5/5/OK/(1971,1939,∠5.0) |
| Patient326/20201117/OD | 2458/23/OK/(2370,1908,∠5.0) | 0/0/cannot_separate_HV | 11/9/OK/(1982,1977,∠-14.0) | 9/9/OK/(1982,1977,∠-14.0) |
| Patient328/20220412/OS | 2246/10/OK/(2740,2212,∠-10.0) | 0/19/cannot_separate_HV | 9/9/OK/(2004,1967,∠-7.0) | 9/9/OK/(2003,1967,∠-7.0) |
| Patient329/20200901/OS | 3070/7/OK/(3196,1622,∠5.0) | 75/0/cannot_separate_HV | 12/12/OK/(2039,2035,∠-15.0) | 11/12/OK/(2042,2034,∠-15.0) |
| Patient329/20250506/OD | 403/25/OK/(3468,1798,∠1.0) | 31/3/OK/(2593,1997,∠6.0) | 4/6/OK/(1970,2011,∠-11.0) | 5/5/OK/(1969,2010,∠-11.0) |
| Patient330/20200512/OD | 2211/4/OK/(3447,2135,∠-6.0) | 285/7/OK/(2239,1937,∠-9.0) | 8/8/OK/(1983,1967,∠-12.0) | 9/7/OK/(1983,1967,∠-12.0) |
| Patient337/20220712/OD | 1427/155/oob(4039.0,2746.0) | 0/0/cannot_separate_HV | 6/6/OK/(1911,1951,∠-9.0) | 7/6/OK/(1909,1952,∠-9.0) |
| Patient339/20250218/OD | 290/126/OK/(3308,1386,∠-20.0) | 0/0/no_crosshair_lines | 12/12/OK/(2014,1971,∠-2.1) | 12/12/OK/(2009,1970,∠-2.1) |
| Patient344/20211005/OD | 2969/2/OK/(3571,2217,∠5.0) | 10/0/cannot_separate_HV | 5/4/OK/(1980,1951,∠-4.0) | 4/4/OK/(1981,1953,∠-4.0) |
| Patient348/20251028/OD | 3254/7/OK/(3593,1697,∠-4.0) | 0/71/cannot_separate_HV | 13/12/OK/(1962,1949,∠-11.1) | 11/11/OK/(1963,1950,∠-11.1) |
| Patient349/20260326/OS | 2422/160/OK/(508,1789,∠-6.0) | 0/186/cannot_separate_HV | 10/10/OK/(2025,2021,∠-9.0) | 10/10/OK/(2027,2020,∠-9.1) |
| Patient35/20210223/OS | 2313/0/cannot_separate_HV | 0/490/cannot_separate_HV | 7/7/OK/(2027,2000,∠-25.0) | 7/6/OK/(2026,1999,∠-25.0) |
| Patient351/20221108/OD | 1919/66/OK/(650,2098,∠5.0) | 0/0/no_crosshair_lines | 8/8/OK/(1951,1978,∠-4.0) | 7/7/OK/(1951,1978,∠-4.0) |
| Patient352/20220607/OD | 2611/91/OK/(3336,2111,∠15.0) | 0/15/cannot_separate_HV | 5/5/OK/(1937,1982,∠-1.0) | 6/7/OK/(1937,1983,∠-1.0) |
| Patient357/20230214/OS | 2473/2/OK/(661,2409,∠-11.0) | 0/40/cannot_separate_HV | 13/13/OK/(2021,1955,∠-9.0) | 14/10/OK/(2025,1958,∠-9.1) |
| Patient36/20220125/OD | 2087/58/OK/(3179,2467,∠7.0) | 60/2/OK/(2197,2127,∠0.0) | 8/6/OK/(1989,1920,∠-7.0) | 8/8/OK/(1990,1920,∠-7.0) |
| Patient360/20250422/OD | 2064/56/OK/(3286,2351,∠15.0) | 0/8/cannot_separate_HV | 12/12/OK/(1949,1931,∠-3.0) | 10/10/OK/(1951,1930,∠-2.9) |
| Patient364/20230530/OD | 2544/13/OK/(689,2354,∠-7.0) | 1/13/OK/(3335,1492,∠-6.9) | 12/10/OK/(1968,1979,∠-16.1) | 9/11/OK/(1968,1983,∠-16.0) |
| Patient37/20210831/OS | 2537/6/OK/(3528,2151,∠0.0) | 1899/134/OK/(3269,1907,∠-11.0) | 3/3/OK/(1985,2031,∠0.0) | 3/3/OK/(1984,2032,∠0.0) |
| Patient37/20240326/OD | 2409/216/OK/(3378,2009,∠-9.0) | 2435/81/OK/(3909,2545,∠20.0) | 3/3/OK/(1921,1969,∠0.0) | 3/3/OK/(1922,1969,∠0.0) |
| Patient38/20220215/OD | 2438/23/OK/(3564,2130,∠8.0) | 1704/14/OK/(841,2168,∠-11.0) | 3/3/OK/(1976,1963,∠0.0) | 3/3/OK/(1976,1962,∠0.0) |
| Patient40/20220315/OS | 2317/8/OK/(3205,2394,∠19.0) | 845/0/cannot_separate_HV | 8/9/OK/(2001,1969,∠-10.0) | 8/9/OK/(2001,1969,∠-10.0) |
| Patient55/20210316/OS | 2636/31/OK/(3126,2311,∠12.0) | 0/0/cannot_separate_HV | 8/8/OK/(1987,1947,∠4.0) | 9/9/OK/(1987,1947,∠4.0) |
| Patient56/20210309/OD | 2125/361/OK/(3442,2076,∠-4.0) | 0/0/no_crosshair_lines | 8/8/OK/(2002,2046,∠-18.0) | 6/6/OK/(2003,2047,∠-18.0) |
| Patient57/20230809/OD | 2797/146/OK/(3192,1657,∠-5.0) | 0/33/cannot_separate_HV | 3/3/OK/(1983,1944,∠0.0) | 3/3/OK/(1983,1944,∠0.0) |
| Patient58/20230627/OS | 2104/60/OK/(490,1852,∠4.0) | 136/9/OK/(2339,2401,∠-24.0) | 3/3/OK/(2007,1955,∠0.0) | 3/3/OK/(2009,1956,∠0.0) |
| Patient63/20231128/OS | 855/141/OK/(677,2265,∠-15.0) | 143/23/OK/(691,1002,∠1.1) | 3/3/OK/(2007,2046,∠0.0) | 3/3/OK/(2007,2046,∠0.0) |
| Patient65/20100922/OS | 2011/42/OK/(683,1886,∠1.0) | 128/4/OK/(468,2603,∠-14.0) | 3/3/OK/(2034,2020,∠0.0) | 3/3/OK/(2033,2020,∠0.0) |
| Patient69/20230808/OD | 3570/123/OK/(3447,2396,∠9.0) | 551/13/OK/(3475,1915,∠-3.0) | 7/6/OK/(1940,1998,∠-7.0) | 7/5/OK/(1941,1999,∠-7.0) |
| Patient70/20210202/OD | 2739/201/OK/(799,2169,∠-14.0) | 0/640/cannot_separate_HV | 3/3/OK/(1734,1941,∠0.0) | 3/3/OK/(1735,1941,∠0.0) |
| Patient70/20230815/OD | 2640/172/OK/(468,1662,∠14.0) | 0/194/cannot_separate_HV | 3/3/OK/(1989,1977,∠0.0) | 3/3/OK/(1991,1977,∠0.0) |
| Patient76/20240409/OS | 2450/120/OK/(2842,1701,∠-10.0) | 0/72/cannot_separate_HV | 3/3/OK/(1995,1973,∠0.0) | 3/3/OK/(1997,1975,∠0.0) |
| Patient77/20240604/OD | 2220/161/OK/(1012,1557,∠19.0) | 0/23/cannot_separate_HV | 3/3/OK/(1994,1999,∠0.0) | 3/3/OK/(1993,2000,∠0.0) |
| Patient80/20230919/OS | 2484/17/OK/(555,2211,∠-2.0) | 10/12/OK/(2177,1909,∠-5.9) | 3/3/OK/(2025,1961,∠0.0) | 3/3/OK/(2027,1960,∠0.0) |
| Patient80/20240723/OD | 2103/243/OK/(3171,2306,∠12.0) | 1/21/OK/(1959,2210,∠-11.9) | 3/3/OK/(1948,1988,∠0.0) | 3/3/OK/(1948,1988,∠0.0) |
| Patient80/20251021/OD | 2062/404/OK/(1260,2308,∠-20.0) | 0/2/cannot_separate_HV | 11/11/OK/(1986,1885,∠10.0) | 12/10/OK/(1979,1887,∠9.9) |
| Patient81/20250819/OD | 2272/144/OK/(1079,2338,∠-10.0) | 0/33/cannot_separate_HV | 3/3/OK/(1963,1949,∠0.0) | 3/3/OK/(1961,1951,∠0.0) |
| Patient83/20221229/OS | 2048/95/OK/(2720,2128,∠14.0) | 12/3/OK/(2398,2176,∠-10.0) | 14/13/OK/(2010,2011,∠-19.9) | 12/13/OK/(2008,2008,∠-20.0) |
| Patient85/20251125/OD | 1250/117/OK/(3137,1726,∠-25.0) | 2103/178/OK/(2235,1604,∠-3.0) | 8/8/OK/(1973,2016,∠-17.0) | 8/8/OK/(1973,2016,∠-17.0) |
| Patient86/20210406/OD | 3162/151/OK/(3343,1765,∠-4.0) | 0/115/cannot_separate_HV | 3/3/OK/(1955,2000,∠0.0) | 3/3/OK/(1954,1998,∠0.0) |
| Patient86/20250819/OS | 2621/80/OK/(3403,1998,∠-1.0) | 0/59/cannot_separate_HV | 3/3/OK/(2036,1990,∠0.0) | 3/3/OK/(2038,1990,∠0.0) |
| Patient87/20230606/OD | 2130/10/OK/(3633,2191,∠-1.0) | 0/0/no_crosshair_lines | 3/3/OK/(1961,1998,∠0.0) | 3/3/OK/(1962,1999,∠0.0) |
| Patient87/20250722/OS | 2480/144/OK/(509,1429,∠9.0) | 207/299/OK/(1290,1090,∠8.0) | 5/5/OK/(2014,1953,∠-12.0) | 5/5/OK/(2015,1953,∠-12.0) |
| Patient88/20250107/OD | 1919/54/OK/(842,1730,∠10.0) | 217/1/OK/(2291,2090,∠-11.0) | 12/10/OK/(1980,1958,∠-10.1) | 12/11/OK/(1976,1958,∠-10.1) |
| Patient92/20251104/OS | 1787/28/OK/(557,1858,∠9.0) | 22/131/OK/(3168,1260,∠5.1) | 8/7/OK/(2039,1963,∠-12.0) | 8/8/OK/(2039,1963,∠-12.0) |
| Patient94/20250204/OS | 2320/79/OK/(3046,2070,∠-13.0) | 0/17/cannot_separate_HV | 7/8/OK/(1997,1939,∠-16.0) | 7/7/OK/(1997,1938,∠-16.0) |
| Patient95/20250401/OD | 3188/393/OK/(934,1313,∠5.0) | 0/568/cannot_separate_HV | 0/0/cannot_separate_HV | 0/0/cannot_separate_HV |
| Patient95/20250812/OD | 79/594/OK/(2896,960,∠-6.0) | 48/330/OK/(3003,1112,∠0.0) | 0/0/cannot_separate_HV | 0/0/cannot_separate_HV |

## Section 3 — Are the accepted FP detections anatomically plausible?

Population: all **715** sidecars with `fovea_fallback == false`. Coordinates from sidecar; intensity from matching `cleaned/*.npy` (same OD-standardized space).

- Distance from image center (px): min **94.9**, median **1430.4**, max **2509.3**
- `(cx,cy)` outside retinal content mask: **231** / 715
- Within 200px of any frame edge: **27** / 715
- Local 100×100 window mean > image content mean (bright tissue): **154** / 715
- Any of the three criteria (implausible): **390** / 715 (**54.5%**)
- Normalised `(cx/W, cy/H)` medians: (**0.656**, **0.510**); ranges cx [0.000, 0.999], cy [0.043, 0.913]

### 10×10 histogram of normalised (cx, cy) — rows = cx bins [0–0.1)…[0.9–1.0], columns = cy bins

```
cx\\cy | 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
0.0   |    0    0    3    3    7    9   10    2    1    1
0.1   |    0    0    2   20   62   78   53    7    2    0
0.2   |    1    0    4   18   17   22    7    1    0    0
0.3   |    0    0    2    2    2    3    1    0    0    0
0.4   |    0    0    0    1    1    3    1    0    0    0
0.5   |    0    0    0    2    2    4    2    0    0    0
0.6   |    0    0    0    2    3    4    4    0    0    0
0.7   |    0    0    3   14   27   19   12    3    0    0
0.8   |    0    1    1   24   73   78   33   13    1    0
0.9   |    0    1    1   10   17    8    4    3    0    0
```

Mass concentrates at **left and right mid-height periphery** (cx bins 0.1–0.2 and 0.8–0.9, cy 0.4–0.6), not at the image center — consistent with spurious Hough intersections on Optos noise rather than foveae.

## Section 4 — FA crosshair as reference

- Visit-eyes with `_FA_0001`: **789**
- TIGHT detection success: **768** (97.34%)
- FA TIGHT errors: `{'OK': 768, 'cannot_separate_HV': 20, 'not_enough_yellow': 1}`
- Pairs with accepted FP sidecar (non-fallback) **and** successful FA TIGHT: **692**
- Euclidean FP–FA `(cx,cy)` distance (px): min **91.1**, median **1441.8**, max **2426.2**

Distance histogram:

| bin (px) | count |
|---|---:|
| 0-50 | 0 |
| 50-100 | 2 |
| 100-200 | 2 |
| 200-400 | 7 |
| 400-800 | 17 |
| 800-1600 | 512 |
| 1600-3200 | 152 |
| >3200 | 0 |

The median distance (**1442 px**) is large (≈36% of a 4000px frame). **Stored FP detections are wrong relative to the FA crosshair.**

## Section 5 — Inpainting damage

Sample: **30** random FPs (seed=42). Compare raw FP (OS-flipped) vs cached `cleaned/*.npy`.

| metric | min | median | max |
|---|---:|---:|---:|
| fraction content px with any-channel \|Δ\|>5 | 0.2292 | 0.6292 | 0.8909 |
| mean abs diff over content | 6.125 | 14.264 | 39.393 |
| PSNR (content, dB) | 13.219 | 19.994 | 27.955 |
| SSIM (512px grayscale) | 0.7232 | 0.9170 | 0.9675 |

Median **~63%** of content pixels are altered by >5 in a channel — consistent with LOOSE mask covering ~63% of FP content (Section 1) and `remove_yellow_overlay` inpainting that mask.

Side-by-side QC PNGs (1000px) in `docs/geometry_audit_images/inpaint/`:

- `01_P36_20200601_OD.png`
- `02_P37_20210525_OD.png`
- `03_P43_20210323_OD.png`
- `04_P55_20210316_OD.png`
- `05_P58_20251014_OS.png`
- `06_P324_20251125_OD.png`
- `07_P328_20201124_OD.png`
- `08_P331_20251014_OD.png`
- `09_P343_20200325_OS.png`
- `10_P360_20250708_OS.png`

## Section 6 — Visual QC

25 random non-fallback visit-eyes (seed=7). Each figure: left = cleaned FP with stored red fovea + 10 zone boundaries; green = FA TIGHT fovea when available; right = FA `_0001` with its TIGHT detection.
Images: `docs/geometry_audit_images/zones/`.

**Assessment method:** visual inspection of each PNG (red vs anatomical macula / FA crosshair). Automated “central-dark” heuristics alone are insufficient — several red dots land in darkish periphery yet are still far from the true fovea (FA green, ~1.1–2.0k px away).

| # | filename | red-dot anatomically plausible fovea? | FA det | FP–FA dist (px) | notes |
|---:|---|---|---|---:|---|
| 1 | `01_P80_20250715_OD.png` | **NO** | OK | 1337 | far_from_FA_crosshair(1337px) |
| 2 | `02_P46_20211130_OS.png` | **NO** | OK | 1565 | far_from_FA_crosshair(1565px); far_from_center(1536px) |
| 3 | `03_P87_20220927_OS.png` | **NO** | OK | 1085 | far_from_FA_crosshair(1085px); bright_tissue(local=43.7>content=42.4) |
| 4 | `04_P352_20210223_OS.png` | **NO** | OK | 1283 | far_from_FA_crosshair(1283px) |
| 5 | `05_P27_20220329_OS.png` | **NO** | OK | 1473 | far_from_FA_crosshair(1473px); far_from_center(1453px) |
| 6 | `06_P35_20201124_OS.png` | **NO** | cannot_separate | n/a | outside_retinal_content; far_from_center(1549px) |
| 7 | `07_P321_20251014_OD.png` | **NO** | OK | 1602 | far_from_FA_crosshair(1602px); outside_retinal_content; far_from_center(1569px) |
| 8 | `08_P37_20240326_OS.png` | **NO** | OK | 1474 | far_from_FA_crosshair(1474px); outside_retinal_content; far_from_center(1553px) |
| 9 | `09_P85_20250107_OS.png` | **NO** | OK | 1375 | far_from_FA_crosshair(1375px) |
| 10 | `10_P331_20250610_OS.png` | **NO** | OK | 1498 | far_from_FA_crosshair(1498px); far_from_center(1488px); bright_tissue(local=119.5>content=61.6) |
| 11 | `11_P29_20260120_OD.png` | **NO** | OK | 1564 | far_from_FA_crosshair(1564px); outside_retinal_content; far_from_center(1547px) |
| 12 | `12_P308_20250916_OS.png` | **NO** | OK | 1582 | far_from_FA_crosshair(1582px); far_from_center(1575px) |
| 13 | `13_P58_20220607_OS.png` | **NO** | OK | 1312 | far_from_FA_crosshair(1312px) |
| 14 | `14_P24_20241008_OD.png` | **NO** | OK | 1450 | far_from_FA_crosshair(1450px); outside_retinal_content; far_from_center(1479px) |
| 15 | `15_P37_20211109_OS.png` | **NO** | OK | 1501 | far_from_FA_crosshair(1501px); outside_retinal_content; far_from_center(1547px) |
| 16 | `16_P92_20250617_OD.png` | **NO** | OK | 1308 | far_from_FA_crosshair(1308px) |
| 17 | `17_P89_20250121_OS.png` | **NO** | OK | 1514 | far_from_FA_crosshair(1514px); outside_retinal_content; far_from_center(1514px) |
| 18 | `18_P33_20210630_OD.png` | **NO** | OK | 1370 | far_from_FA_crosshair(1370px); outside_retinal_content; far_from_center(1441px) |
| 19 | `19_P62_20240813_OD.png` | **NO** | OK | 1224 | far_from_FA_crosshair(1224px); bright_tissue(local=28.6>content=28.0) |
| 20 | `20_P37_20230725_OD.png` | **NO** | OK | 2002 | far_from_FA_crosshair(2002px); far_from_center(1798px); bright_tissue(local=117.5>content=87.5) |
| 21 | `21_P327_20251209_OD.png` | **NO** | OK | 1100 | far_from_FA_crosshair(1100px); bright_tissue(local=53.3>content=46.8) |
| 22 | `22_P91_20250107_OS.png` | **NO** | OK | 1304 | far_from_FA_crosshair(1304px) |
| 23 | `23_P30_20210427_OS.png` | **NO** | OK | 1403 | far_from_FA_crosshair(1403px) |
| 24 | `24_P328_20250107_OS.png` | **NO** | OK | 1585 | far_from_FA_crosshair(1585px); outside_retinal_content; far_from_center(1572px) |
| 25 | `25_P41_20210713_OS.png` | **NO** | OK | 1535 | far_from_FA_crosshair(1535px); far_from_center(1445px) |

**Summary:** **0 / 25** red dots are anatomically plausible fovea positions. In every panel where FA TIGHT succeeded, the green dot sits on the visible yellow FA crosshair at the macula; the red stored FP point is displaced to the mid-peripheral retina and the zone grid is centered there.

## Section 7 — Blast radius

Criteria for “implausible” = Section 3 any-of (outside content ∪ near edge ∪ bright tissue) on **non-fallback** accepted FP detections. Fallback visits counted separately (known center-geometry).

- Total training rows: **7720** from **772** images
- Rows from implausible accepted FP detections: **3900** (50.52%)
- Rows from fallback visits: **570**
- Rows OK by Section 3: **3250** (42.10%)
- Implausible ∪ fallback: **4470** (57.90%)
- Images: implausible **390**, fallback **57**, OK **325**

### By canonical_split

| split | implausible_s3 rows | fallback rows | ok rows |
|---|---:|---:|---:|
| train | 1860 | 230 | 1630 |
| val | 420 | 30 | 300 |
| test | 390 | 90 | 360 |
| outside_canonical | 1230 | 220 | 960 |

### By Zone_Label tier

| status | tier0 | tier1 | tier2 |
|---|---:|---:|---:|
| implausible_s3 | 2471 | 543 | 886 |
| fallback | 368 | 68 | 134 |
| ok | 1931 | 560 | 759 |

Non-exclusive flag row counts among implausible_s3: outside_content=2310, near_edge=270, bright_tissue=1540.

