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
    arguments = parser.parse_args()
    return arguments.__dict__


def create_eda() -> None:
    params = dict(filter(lambda x: x[1] is not None, get_params().items()))
    df = pd.read_csv(params["path_to_dataset"])
    profile = ProfileReport(
        df, title="Pandas Profiling Report Forest Dataset", explorative=True
    )
    profile.to_file(output_file="eda/eda_forest_dataset.html")
