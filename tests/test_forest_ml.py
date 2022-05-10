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
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--path-to-dataset'" in result.output


def test_error_for_invalid_use_scaler(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--use-scaler",
            "52"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--use-scaler'" in result.output


def test_error_for_invalid_model(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--model",
            "SuperDuperModel"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output


def test_error_for_invalid_fetengtech(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--fetengtech",
            "SuperDuperFeatureEngineering"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--fetengtech'" in result.output


def test_error_for_invalid_max_depth(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--max-depth",
            -1
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-depth'" in result.output

def test_error_for_invalid_n_estimators(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--n-estimators",
            -1
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--n-estimators'" in result.output

def test_error_for_invalid_max_features(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--max-features",
            "super-max-feature"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-features'" in result.output

def test_error_for_invalid_n_neighbors(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--n-neighbors",
            -1
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--n-neighbors'" in result.output

def test_error_for_invalid_weights(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--weights",
            "super-weight"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--weights'" in result.output

def test_error_for_max_iter(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--max-iter",
            "super-max-iter"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-iter'" in result.output

def test_error_for_c(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--C",
            "super-c"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--C'" in result.output

def test_error_for_penalty(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--penalty",
            "super-penalty"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--penalty'" in result.output

def test_error_for_solver(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--path-to-dataset",
            r"D:/dev/rsschool-ml-capstone-project/data/train.csv",
            "--solver",
            "super-solver"
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--solver'" in result.output


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
