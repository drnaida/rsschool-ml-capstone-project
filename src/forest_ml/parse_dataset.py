from pathlib import Path

import pandas as pd
from typing import Tuple


def get_dataset(
    csv_path: Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    print(type(features), type(target))
    return features, target
