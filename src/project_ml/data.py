import pandas as pd
from typing import Tuple
from pathlib import Path
from pandas_profiling import ProfileReport

def get_dataset(csv_path: Path, eda:bool) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    print(dataset.head())
    # click.echo(f"Dataset shape: {dataset.shape}.")



    if eda:
        profile = ProfileReport(dataset, minimal=True)
        profile.to_file("output.html")
    return  features, target


