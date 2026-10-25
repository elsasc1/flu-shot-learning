# Flu Shot Learning — DrivenData

Predicting whether people received the **H1N1** and **seasonal flu** vaccines, from the U.S. National 2009 H1N1 Flu Survey.

**Public leaderboard: ROC AUC 0.8636 — ranked 122nd out of 8,600+ participants (top 1.5%)** · October 2026
Competition page: [drivendata.org/competitions/66/flu-shot-learning](https://www.drivendata.org/competitions/66/flu-shot-learning/)

## The problem

- 26,707 survey respondents, 35 features: concern and knowledge about the flu, preventive behaviours, doctor recommendations, opinions on vaccine effectiveness and risk, demographics, employment.
- Many missing values: up to 50% for employment fields, 46% for health insurance.
- Two binary targets: `h1n1_vaccine` (21% positive) and `seasonal_vaccine` (47% positive).
- Metric: ROC AUC of each target, averaged.

## Approach

1. **Validation.** Stratified 5-fold cross-validation, the same folds for every model. Every model produces out-of-fold (OOF) predictions, which are used both to score it and to decide how to blend.
2. **Baseline.** Logistic regression in a scikit-learn pipeline: median imputation with missing-value indicators, standard scaling, one-hot encoding. The pipeline is refit inside each fold to avoid leakage.
3. **Gradient boosting.**
   - LightGBM with native categorical features and early stopping.
   - CatBoost (ordered target statistics), in two variants: categorical columns only, and every survey answer treated as a category.
4. **Blending.** I compared a simple average, rank averaging with grid-searched weights, and cross-target stacking. The simple average scored as well as the others with less risk of overfitting the OOF predictions, so I kept it.
5. **Final predictions.** Each model was refit on 100% of the data, with a fixed number of trees (mean best iteration from CV × 1.15). Predictions were averaged over several seeds: 5 for LightGBM, 3 for each CatBoost variant.

## Results

| Model | CV ROC AUC (OOF) | Public leaderboard |
|---|---|---|
| Logistic regression (baseline) | 0.8587 | – |
| LightGBM | 0.8688 | – |
| CatBoost, numeric | 0.8685 | – |
| CatBoost, all-categorical | 0.8686 | – |
| **v1**: LightGBM + CatBoost, test = average of the 5 fold models | 0.8696 | 0.8630 (216th) |
| **v2**: 3-model blend, full-data refit, seed averaging | 0.8697 | **0.8636 (122nd)** |

The refit and the seed averaging only change the test predictions. CV cannot measure them, but the leaderboard moved in the same direction as CV.

## Insights

- **The doctor's recommendation is the strongest signal.** When the doctor recommended the H1N1 vaccine, 53% of people got it, against 14% when they didn't. It is the top feature of the H1N1 model.
- **Missing values carry information.** About 88% of respondents with no employment industry are unemployed or not in the labour force, and most of the rest did not report their status. I kept "missing" as its own category instead of imputing.
- **The two vaccines are linked.** 83% of H1N1-vaccinated respondents also got the seasonal vaccine.

## What didn't work

- **15 hand-crafted interaction features**, such as doctor recommendation × perceived effectiveness, or industry × occupation. With all 15, CV AUC dropped by 0.0011; without industry × occupation, they were neutral (−0.00003), so I dropped them. Trees already learn these interactions, and the industry × occupation cross creates many tiny categories that overfit.
- **Optimised blend weights and cross-target stacking** gave no gain over a plain average.

## Reproduce

Data is not included: the competition rules forbid redistributing it. Download the 4 CSV files from the competition page into `data/`.

```bash
pip install -e .                                        # Python >= 3.10
python -m flushot train lgbm cat_num cat_allcat         # CV + full-data refit, saves outputs/preds_*.npz
python -m flushot baseline                              # logistic regression baseline
python -m flushot compare                               # OOF AUC of every model combination
python -m flushot submit lgbm cat_num cat_allcat        # v2 submission, format checked before writing
```

The package reproduces the original competition scripts exactly: run on the same data, both give identical out-of-fold and test predictions (max absolute difference 0.0).

On a Mac, if LightGBM complains about `libomp.dylib`, run `brew install libomp`.

## Tests

The real data cannot be shared, so the tests run on **synthetic data with the exact schema** of the competition files (`python -m flushot fake-data`). GitHub Actions runs the linter, the tests and a full smoke run on every push.

```bash
pip install -e ".[dev]"
pytest
```

What the tests check:

- **No leakage:** every row gets exactly one out-of-fold prediction, from a model that never saw it; the baseline's imputer and scaler are fitted on training rows only.
- **Validation:** folds partition the data, keep the class balance and are identical for a given seed.
- **Reproducibility:** two runs with the same seeds give bit-identical predictions.
- **Submission:** wrong columns, reordered or missing IDs, missing values and probabilities outside [0, 1] are all rejected before writing the file.
- **End to end:** fake data → train → compare → submit, through the command line.

## Project structure

| Path | Role |
|---|---|
| `src/flushot/data.py` | loading, the two kept features, categorical encoding |
| `src/flushot/models.py` | LightGBM and CatBoost configurations of the v2 submission |
| `src/flushot/cv.py` | stratified 5-fold CV, OOF predictions, full-data refit with seed averaging |
| `src/flushot/baseline.py` | logistic regression baseline in a leak-free scikit-learn pipeline |
| `src/flushot/submission.py` | blending, combination ranking, submission checks |
| `src/flushot/synthetic.py` | synthetic data with the real schema, for tests and CI |
| `src/flushot/cli.py` | `python -m flushot` commands |
| `tests/` | pytest suite |
| `experiments/` | the original competition scripts, kept as they were run (v1, blending and stacking experiments, feature-engineering A/B test); run them from the repo root, e.g. `python experiments/train2.py lgbm` |

## Data citation

DrivenData. (2020). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved October 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning.
