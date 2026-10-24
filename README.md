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
pip install pandas numpy scipy scikit-learn lightgbm catboost
python train.py                                     # v1: CV of logistic regression, LightGBM, CatBoost
python blend.py                                     # blending experiments (average / rank weights / stacking)
python make_submission_v1.py                        # submission 1 (run after blend.py, which also writes submission.csv)
python train2.py lgbm                               # v2: CV + full-data refit with seeds
python train2.py cat_allcat
python train2.py cat_num
python compare.py write lgbm2+allcat2+catnum2       # all blend combinations + submission 2
python fe_test.py                                   # feature-engineering A/B test
```

On a Mac, if LightGBM complains about `libomp.dylib`, run `brew install libomp`.

| File | Role |
|---|---|
| `train.py` | v1: 5-fold CV of the three models; test predictions averaged over fold models |
| `blend.py` | blending experiments |
| `make_submission_v1.py` | 50/50 LightGBM + CatBoost submission, with format checks |
| `train2.py` | v2: CV, then full-data refit averaged over several seeds |
| `compare.py` | OOF score of every model combination; writes submission 2 |
| `fe_test.py` | A/B test of the engineered features |

## Data citation

DrivenData. (2020). *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines.* Retrieved October 2026 from https://www.drivendata.org/competitions/66/flu-shot-learning.
