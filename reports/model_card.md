# Model card - breast cancer survival models

_Generated automatically by `run_pipeline.py` on 2026-09-25._

## Intended use
Research and teaching: comparing survival models on a public SEER breast cancer extract.
**Not** validated for clinical decision-making.

## Data
- 4023 patients after cleaning, 616 deaths (15.3%); the rest are censored.
- Split (stratified by stage and outcome): train 2816, validation 603, test 604.
- Features (13): `age`, `race`, `marital_status`, `t_stage`, `n_stage`, `stage_6th`, `grade`, `a_stage`, `tumor_size`, `estrogen_status`, `progesterone_status`, `regional_node_examined`, `regional_node_positive`

## Models
Hyper-parameters were chosen by Uno's C-index on the validation split; test was used only once.

- **AJCC stage only (baseline)** - best params `{}`, validation C = 0.664
- **Cox PH (ridge)** - best params `{'alpha': 0.1}`, validation C = 0.760
- **Random Survival Forest** - best params `{'min_samples_leaf': 15, 'max_features': 0.5}`, validation C = 0.737
- **Gradient Boosted Cox** - best params `{'learning_rate': 0.1, 'max_depth': 3}`, validation C = 0.741

## Test-set results

| Model | Uno's C (95% CI) | Harrell's C | IBS (95% CI) | IBS skill vs KM | AUC 12m | AUC 36m | AUC 60m | AUC 96m |
|---|---|---|---|---|---|---|---|---|
| AJCC stage only (baseline) | 0.658 (0.602-0.719) | 0.663 | 0.095 (0.079-0.112) | +5.3% | 0.726 | 0.692 | 0.633 | 0.646 |
| Cox PH (ridge) | 0.727 (0.681-0.786) | 0.738 | 0.089 (0.074-0.105) | +11.3% | 0.824 | 0.780 | 0.729 | 0.701 |
| Random Survival Forest | 0.708 (0.658-0.764) | 0.724 | 0.090 (0.075-0.106) | +10.9% | 0.827 | 0.775 | 0.713 | 0.651 |
| Gradient Boosted Cox | 0.724 (0.679-0.779) | 0.730 | 0.089 (0.073-0.106) | +11.5% | 0.826 | 0.782 | 0.715 | 0.685 |

Best model (chosen by validation Uno's C): **Cox PH (ridge)** (C = 0.727, IBS = 0.089). Evaluation horizons: 12, 36, 60, 96 months.

Most important variables (permutation, Cox PH (ridge)): grade, regional_node_positive, progesterone_status, estrogen_status, n_stage.

## Limitations
- Single registry extract (SEER, diagnoses 2006-2010); no external validation.
- Follow-up is capped at about 9 years, so long-term risk is extrapolated.
- The outcome is all-cause death, not breast-cancer-specific death.
- No treatment, HER2, or comorbidity data; important predictors are missing.
- Confidence intervals come from bootstrapping the test set only (models are not refitted).
