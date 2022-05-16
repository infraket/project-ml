import pandas as pd
from typing import Tuple
from pathlib import Path
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

def get_dataset(csv_path: Path, random_state: int, test_split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    X_train, X_test, Y_train, Y_test = train_test_split(features, target,
                                                        test_size=test_split_ratio, random_state=random_state)
    print(dataset.head())





    return  X_train, X_test, Y_train, Y_test
def eda_profile(csv_path: Path,eda:bool):
    dataset = pd.read_csv(csv_path)
    if eda:
        profile = ProfileReport(dataset, minimal=True)
        profile.to_file("output.html")


