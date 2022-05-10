from click.testing import CliRunner
import pytest
import click
import os
import pathlib
from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_random_state(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--random-state",
            -1,
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--path-save-model",
            r"D:/dev/rsschool-ml-capstone-project/data/model.joblib",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output


def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            10,
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--path-save-model",
            r"D:/dev/rsschool-ml-capstone-project/data/model.joblib",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output

def test_error_for_invalid_dataset_path(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.sv",
            "--path-save-model",
            r"D:/dev/rsschool-ml-capstone-project/data/model.joblib",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--path-to-dataset'" in result.output


def test_valid_parameters(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    cwd = os.getcwd()
    p = pathlib.Path(cwd)
    path_to_dataset = str(p) + "/tests/test_sample.csv"
    path_to_save_model = str(p) + "/data/model.joblib"
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--path-to-dataset",
                path_to_dataset,
                "--cross-validation-type",
                "k-fold",
                "--path-save-model",
                path_to_save_model
            ],
        )
        assert result.exit_code == 0