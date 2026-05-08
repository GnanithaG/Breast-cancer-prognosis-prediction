from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest


def train_rsf_pipeline(
    preprocessor,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state=7,
):
    model = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe