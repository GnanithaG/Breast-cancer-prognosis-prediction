# Model card - breast cancer survival models

_Generated automatically by `run_pipeline.py` on 2026-09-25._

## Intended use
Research and teaching: comparing survival models on a public SEER breast cancer extract.
**Not** validated for clinical decision-making.

## Data
- 4023 patients after cleaning, 616 deaths (15.3%); the rest are censored.
- Split (stratified by stage and outcome): train 2816, validation 603, test 604.
- Features (14): `age`, `race`, `marital_status`, `t_stage`, `n_stage`, `stage_6th`, `differentiation`, `grade`, `a_stage`, `tumor_size`, `estrogen_status`, `progesterone_status`, `regional_node_examined`, `regional_node_positive`

## Models
Hyper-parameters were chosen by Uno's C-index on the validation split; test was used only once.

- **Cox PH (ridge)** - best params `{'alpha': 0.1}`, validation C = 0.760
- **Random Survival Forest** - best params `{'min_samples_leaf': 15, 'max_features': 0.5}`, validation C = 0.739
- **Gradient Boosted Cox** - best params `{'learning_rate': 0.1, 'max_depth': 3}`, validation C = 0.741

## Test-set results

| Model | Uno's C (95% CI) | Harrell's C | IBS (95% CI) | IBS skill vs KM | AUC 12m | AUC 36m | AUC 60m | AUC 96m |
|---|---|---|---|---|---|---|---|---|
| Cox PH (ridge) | 0.727 (0.682-0.786) | 0.738 | 0.089 (0.074-0.105) | +11.3% | 0.824 | 0.780 | 0.729 | 0.701 |
| Random Survival Forest | 0.705 (0.655-0.759) | 0.724 | 0.090 (0.075-0.105) | +10.8% | 0.828 | 0.778 | 0.715 | 0.649 |
| Gradient Boosted Cox | 0.724 (0.679-0.779) | 0.730 | 0.089 (0.073-0.106) | +11.5% | 0.826 | 0.782 | 0.715 | 0.685 |

Best model (chosen by validation Uno's C): **Cox PH (ridge)** (C = 0.727, IBS = 0.089). Evaluation horizons: 12, 36, 60, 96 months.

Most important variables (permutation, Cox PH (ridge)): regional_node_positive, progesterone_status, estrogen_status, n_stage, race.

## Limitations
- Single registry extract (SEER, diagnoses 2006-2010); no external validation.
- Follow-up is capped at about 9 years, so long-term risk is extrapolated.
- The outcome is all-cause death, not breast-cancer-specific death.
- No treatment, HER2, or comorbidity data; important predictors are missing.
- Confidence intervals come from bootstrapping the test set only (models are not refitted).
