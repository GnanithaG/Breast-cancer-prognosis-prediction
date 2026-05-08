from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis


def train_cox_pipeline(
    preprocessor,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state=7,
):
    model = CoxPHSurvivalAnalysis(alpha=0.1)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe