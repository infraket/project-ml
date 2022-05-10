import pandas as pd
from typing import Tuple
from pathlib import Path
from pandas_profiling import ProfileReport

def get_dataset(csv_path: Path, eda:bool) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    print(dataset.head())

    if eda:
        profile = ProfileReport(dataset, minimal=True)
        profile.to_file("output.html")


