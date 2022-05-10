import pandas as pd
from pandas_profiling import ProfileReport
import pathlib
import argparse


def get_params() -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Create the EDA")
    parser.add_argument(
        "--path-to-dataset",
        type=pathlib.Path,
        required=True,
        help="A path to the file with your dataset",
    )
    parser.add_argument(
        "--path-to-save-sample",
        type=pathlib.Path,
        required=True,
        help="A path where to save the sample",
    )
    arguments = parser.parse_args()
    return arguments.__dict__


def create_sample_from_dataset() -> None:
    params = dict(filter(lambda x: x[1] is not None, get_params().items()))
    df = pd.read_csv(params["path_to_dataset"])
    print(df.shape)
    sample = df.sample(n=3000, random_state=42)
    print(sample.shape)
    sample.to_csv(params["path_to_save_sample"], index=False)