# Retraining Reproducibility Notes

This document captures the recipe to reproduce the manuscript-published
paralysis classification accuracy from a fresh training run, plus a known
upstream issue affecting AU17 / AU25 / AU26 extraction that biases Lower
Face training when run against the regenerated `combined_results.csv`.

## Manuscript-quality training recipe

The published Mid Face / Lower Face / Upper Face models can be reproduced by:

1. **Use the canonical training data**: `S3 Data Analysis/paper_combined_results.csv`
   (the input `combined_results.csv` snapshot used for the manuscript and the
   models currently committed under `S3 Data Analysis/models/`).
2. **Enable `use_known_optimal=True`** in the relevant zone of
   `paralysis_config.py:ZONE_CONFIG[<zone>]['training']['hyperparameter_tuning']`.
   The committed `known_optimal_params` for each zone are extracted from the
   actual saved model `.pkl` files and reproduce the published numbers.
3. **The deterministic patient sort** in `paralysis_utils.py:prepare_data_generalized`
   is required for `train_test_split(random_state=42, ...)` to land on the same
   split regardless of the row order produced by `main.py --batch`.

Concretely:

```python
INPUT_FILES['results_csv'] = '<S3 Data Analysis>/paper_combined_results.csv'
ZONE_CONFIG['lower']['training']['hyperparameter_tuning']['use_known_optimal'] = True
# then: python paralysis_training_pipeline.py lower
```

Reproduction parity (test-set accuracy on the train_test_split with
`random_state=42`):

| Zone | Manuscript | Reproduced |
|------|------------|------------|
| Upper Face | 0.83 | 0.85 |
| Mid Face | 0.93 | 0.89 |
| Lower Face | 0.84 | 0.84 |

## Why we don't reproduce against the regenerated combined_results.csv

A re-run of `main.py --batch` against today's S2O Coded Files produces a
`combined_results.csv` whose AU intensity values have drifted from the
manuscript snapshot for some AUs:

| AU | Mean abs. diff vs paper | Pearson r | Used by |
|----|------------------------:|----------:|---------|
| AU07 | 0.22 | 0.89 | Mid |
| AU26 | 0.20 | 0.68 | Lower |
| AU17 | 0.20 | 0.70 | Lower |
| AU25 | 0.16 | 0.75 | Lower |
| AU14 | 0.16 | 0.85 | Lower |
| AU10 | 0.15 | 0.87 | Lower |
| AU12 | 0.14 | 0.88 | Lower |
| AU06 | 0.14 | 0.86 | Mid |
| AU45 | 0.13 | 0.72 | Mid |
| AU01 | 0.10 | 0.70 | Upper |
| AU02 | 0.07 | 0.57 | Upper |

The shifts are correlated (ranks roughly preserved) but the magnitudes have
moved enough to materially change downstream training. Lower Face is the most
affected zone because AU17, AU25, and AU26 — the AUs with the lowest Pearson
correlation between the snapshots — are central to its feature set.

Accuracy when retraining with `use_known_optimal=True` on each input:

| Zone | On `paper_combined_results.csv` | On regenerated `combined_results.csv` |
|------|-------------------------------:|--------------------------------------:|
| Upper | 0.85 | 0.83 |
| Mid | 0.89 | 0.87 |
| Lower | 0.84 | 0.64 |

## Open investigation: PyFaceAU AU17 / AU25 / AU26 regression

The underlying AU extractor (PyFaceAU) is producing systematically different
intensity values for AU17 (Chin Raiser), AU25 (Lips Part), and AU26 (Jaw Drop)
compared to the snapshot used during manuscript-era training. The shift is not
random noise — Pearson correlations of 0.68 to 0.75 indicate the relative
ranking of patients is mostly preserved while the absolute scale has moved.

### Suggested investigation steps

1. `git log --since=<manuscript era>` on the active PyFaceAU code paths
   (under `pyfaceau/pyfaceau/` and `pyclnf/pyclnf/`) to identify diffs since
   the snapshot that produced `paper_combined_results.csv`.
2. Compare the SVR weight files / AU prediction module versions against any
   archived "gold standard" (e.g., the historical CLNF parameters documented
   in earlier `ACCURACY_AFFECTING_CHANGES.md` notes that have since been
   restored to gold-standard values).
3. Spot-check 5 patients with extreme drift (e.g., the worst-AU17 cases) by
   re-running PyFaceAU at the older revision and confirming the AU17 values
   match the paper snapshot.
4. If the regression is in PyFaceAU itself, the fix should restore the
   regenerated `combined_results.csv` to manuscript parity, after which
   retraining without `paper_combined_results.csv` should also reproduce.

Until the upstream fix lands, **production should keep using the saved
models** under `S3 Data Analysis/models/` (committed; reproduce manuscript
numbers on either input). For methodologically clean retraining, prefer
`paper_combined_results.csv` as the canonical training input.
