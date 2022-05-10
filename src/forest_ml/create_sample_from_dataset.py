import pandas as pd
import pathlib
import argparse
import os


def get_params() -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Create the EDA")
    parser.add_argument(
        "--path-to-dataset",
        type=pathlib.Path,
        required=True,
        help="A path to the file with your dataset",
    )
    arguments = parser.parse_args()
    return arguments.__dict__


def create_sample_from_dataset() -> None:
    params = dict(filter(lambda x: x[1] is not None, get_params().items()))
    df = pd.read_csv(params["path_to_dataset"])
    print(df.shape)
    sample = df.sample(n=3000, random_state=42)
    print(sample.shape)
    cwd = os.getcwd()
    p = pathlib.Path(cwd)
    path_to_sample = str(p) + "/tests/test_sample.csv"
    sample.to_csv(path_to_sample, index=False)