# Text-free audit artifact

Companion material for *Concept Recognition Is Not Concept Sufficiency: Exact Macro-F1 Auditing of a Credibility Model for Vietnamese-Language Facebook Posts* (EAI FISAT 2026).

The artifact contains no post text, URLs, page names or account identifiers. It supports parameter and arithmetic inspection and recomputation of all test-set results; it does not support retraining.

## Files

| File | Content |
|---|---|
| `locked_parameters.json` | Everything fixed before test access: concept thresholds, final seeds, encoder recipe, all five CBM aggregators (coefficients to four decimals, intercept, selected C, target threshold; Table 6), the cumulative-intervention order, control settings, and the selection procedure. |
| `results_summary.json` | Aggregate test results: Table 2 (concept recognition), Table 3 (target performance), Table 4 (by recorded C01), Table 5 (cumulative gold substitution, all k), the RQ2 figures, and the post-hoc analyses added in revision. |
| `release_manifest.csv` | The row-level release manifest of Sections 3.1 and 3.3: one row per collected item (8,468 rows) with eligibility, exclusion code, split, near-duplicate component, and, for the 8,069 retained posts, the target and gold concepts. |
| `test_predictions.csv` | One row per test post (1,210 rows). |
| `test_gold_profile_counts.csv` | The 172 distinct gold-concept profiles of the test set with positive and negative counts. |
| `exact_optimum.py` | The exact macro-F1 dynamic programme of Section 4.4. Needs only the profile counts. |
| `verify_results.py` | Recomputes the Table 3 macro-F1 of the hard CBM, TF–IDF and PhoBERT, the gold-profile optimum, and the paired bootstrap intervals of Sections 5.1 and 5.2. |
| `figures/` | Figures 1 and 2 with alt text (`ALT_TEXT.md`). |

## Schema

`test_predictions.csv`

| Column | Type | Meaning |
|---|---|---|
| `post` | int | Row number 0–1209; not linked to any post identifier |
| `y` | 0/1 | Codebook target (1 = codebook-positive, archivally non-credible) |
| `c01`–`c10` | 0/1 | Recorded gold concepts |
| `cbm_probability` | float | Hard CBM: mean of the five path scores |
| `tfidf_probability` | float | TF–IDF logistic regression |
| `phobert_probability` | float | PhoBERT control: mean over seeds 42, 1337, 2024 |

`release_manifest.csv`

| Column | Type | Meaning |
|---|---|---|
| `row_id` | int | Archive row identifier (8,468 collected items) |
| `eligibility` | text | `retained` (8,069) or `excluded` (399) |
| `exclusion_code` | text | For excluded rows: `out_of_scope_writing_system` (377), `generation_prompt_leakage` (13), or `mixed_or_duplicate` (9) |
| `split` | text | `train` (5,649), `val` (1,210) or `test` (1,210) for retained rows |
| `near_duplicate_component` | int | Component id for the 451 retained posts in the 104 non-singleton near-duplicate components (TF–IDF word 1–2-gram cosine ≥ 0.8); empty for singletons. All such components are in `train`. |
| `y` | 0/1 | Codebook target, retained rows only |
| `c01`–`c10` | 0/1 | Recorded gold concepts, retained rows only |

`test_predictions.csv` rows follow the order of test rows in `release_manifest.csv`.

`test_gold_profile_counts.csv`

| Column | Type | Meaning |
|---|---|---|
| `c01`–`c10` | 0/1 | One observed gold-concept profile |
| `n_positive` | int | Test posts with this profile and y = 1 |
| `n_negative` | int | Test posts with this profile and y = 0 |

Decision thresholds, fixed on training out-of-fold predictions: 0.41 (CBM), 0.49 (TF–IDF), 0.18 (PhoBERT).

## Reproduce

```
pip install -r requirements.txt

python verify_results.py --bootstrap 2000
# cbm      macro-F1 = 0.7776
# tfidf    macro-F1 = 0.8694
# phobert  macro-F1 = 0.9058
# gold-profile optimum = 0.8255
# optimum            95% interval [0.8156, 0.8561]
# PhoBERT - optimum  95% interval [0.0465, 0.0923]
# TF-IDF - optimum   95% interval [0.0105, 0.0546]
# CBM - PhoBERT      95% interval [-0.1528, -0.1032]
# CBM - TF-IDF       95% interval [-0.1140, -0.0686]

python exact_optimum.py
# profiles=172  N+=597  N-=613
# optimum macro-F1=0.8255  TP=486  FP=100
```

`exact_optimum.py --bootstrap` resamples from the profile counts alone, so its interval differs from the paired bootstrap in `verify_results.py` by Monte Carlo variation (about 0.001). The paper reports the paired intervals.

## Concept codes

| Code | Concept |
|---|---|
| C01 | Provenance: post from the designated page |
| C02 | Accusatory or insinuating rhetorical question |
| C03 | Sensational wording or capitalisation |
| C04 | Extreme affect or imputation of malice/incompetence |
| C05 | At least two checkable anchors |
| C06 | Vague or unverifiable attribution |
| C07 | Oppression, blame or zero-sum framing |
| C08 | Call to share, boycott or act |
| C09 | Generalisation, misdirection or whataboutism |
| C10 | Opinion or narration dominates neutral reporting |

## License

Code (`*.py`): MIT, see `LICENSE`. Data files (`*.csv`, `*.json`) and figures: CC BY 4.0, see `LICENSE-DATA.md`.

Table numbers refer to the camera-ready version of the paper.

## Citation

Duong, T. M. T., Tran, K. V., Le, A. T., Le, H. T.: Concept Recognition Is Not Concept Sufficiency: Exact Macro-F1 Auditing of a Credibility Model for Vietnamese-Language Facebook Posts. In: EAI FISAT 2026.
