from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def create_pipeline(
        use_scaler: bool, max_iter: int, log_reg_class: bool, pca_scaler: bool, penalty: str, n_neighbors: int,
) -> Pipeline:
    pipeline_steps = []

    if log_reg_class:
        pipeline_steps.append(("scaler", StandardScaler()))
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter, penalty=penalty)

            )
        )
    if use_scaler:
        if pca_scaler:
            pipeline_steps.append(("scaler", PCA()))
        else:
            pipeline_steps.append(("scaler", StandardScaler()))

    else:
        pipeline_steps.append(
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors))
        )

    return Pipeline(steps=pipeline_steps)
