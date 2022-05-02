from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    return X_train, X_test, y_train, y_test