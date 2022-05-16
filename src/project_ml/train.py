from pathlib import Path
import click
import mlflow.sklearn
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from .pipeline import create_pipeline, create_pipeline_rf
from .data import get_dataset, eda_profile
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier



@click.command()
@click.option(

    "-d",
    "--dataset-path",
    default="./data/train.csv",

    type=click.Path(path_type=Path),
    # type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--random-f",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--model-type",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
)
@click.option(
    "--test-split-ratio",
    default=10,
    type=int,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)

@click.option(
    "-eda",
    "--eda-profiling",
    type=bool,
    default=False, )
def train(
    dataset_path: Path,
    save_model_path: Path,
    model_type: bool,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    random_f: bool,
    eda_profiling: bool,
) -> None:
    X_train, X_test, Y_train, Y_test = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )

    eda_profile(dataset_path, eda_profiling)

    with mlflow.start_run():
        if model_type:
            pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state,)
            pipeline.fit(X_train, Y_train)
            mlflow.log_param("Model",  "LogisticRegression")

        else:
            pipeline = create_pipeline_rf(random_f, )
            pipeline.fit(X_train, Y_train)
            mlflow.log_param("Model", "Random Forest Classifier")

        accuracy = accuracy_score(Y_test, pipeline.predict(X_test))
        mlflow.log_param("random_f", random_f)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")