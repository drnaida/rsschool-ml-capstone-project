from click.testing import CliRunner
import pytest
import click

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
            "--cross-validation-type",
            "k-fold",
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
            "--cross-validation-type",
            "k-fold",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output


def test_valid_parameters(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--path-to-dataset",
                "D:/dev/rsschool-ml-capstone-project/data/train.csv",
                "--cross-validation-type",
                "k-fold",
                "--path-save-model",
                "D:/dev/rsschool-ml-capstone-project/data/model.joblib"
            ],
        )
        assert result.exit_code == 0