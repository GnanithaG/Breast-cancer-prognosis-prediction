# Hazard ratios - Cox PH (ridge)

Categorical levels vs the most common level (reference). Numeric variables per 1 SD. Ridge-penalised, so estimates are shrunk slightly towards 1.

| variable               | level                |   hazard_ratio |
|:-----------------------|:---------------------|---------------:|
| age                    | per SD (+8.9)        |           1.24 |
| tumor_size             | per SD (+21.3)       |           1.04 |
| regional_node_examined | per SD (+8.1)        |           0.76 |
| regional_node_positive | per SD (+5.1)        |           1.40 |
| race                   | Black                |           1.33 |
| race                   | Other                |           0.72 |
| race                   | White (reference)    |           1.00 |
| marital_status         | Divorced             |           1.34 |
| marital_status         | Married (reference)  |           1.00 |
| marital_status         | Separated            |           1.74 |
| marital_status         | Single               |           1.20 |
| marital_status         | Widowed              |           1.17 |
| t_stage                | T1                   |           0.78 |
| t_stage                | T2 (reference)       |           1.00 |
| t_stage                | T3                   |           1.17 |
| t_stage                | T4                   |           1.58 |
| n_stage                | N1 (reference)       |           1.00 |
| n_stage                | N2                   |           1.64 |
| n_stage                | N3                   |           1.48 |
| stage_6th              | IIA (reference)      |           1.00 |
| stage_6th              | IIB                  |           1.26 |
| stage_6th              | IIIA                 |           1.05 |
| stage_6th              | IIIB                 |           1.47 |
| stage_6th              | IIIC                 |           1.33 |
| grade                  | 1                    |           0.67 |
| grade                  | 2 (reference)        |           1.00 |
| grade                  | 3                    |           1.40 |
| grade                  | 4                    |           2.81 |
| a_stage                | Distant              |           1.06 |
| a_stage                | Regional (reference) |           1.00 |
| estrogen_status        | Negative             |           1.76 |
| estrogen_status        | Positive (reference) |           1.00 |
| progesterone_status    | Negative             |           1.58 |
| progesterone_status    | Positive (reference) |           1.00 |
