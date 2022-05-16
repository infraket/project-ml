from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(
        use_scaler: bool, max_iter: int, logreg_c: float, random_state: int,
) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_c
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_rf(random_f: bool, ) -> Pipeline:
    pipeline_steps = []
    if random_f:
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(),
            )
        )
        return Pipeline(steps=pipeline_steps)
