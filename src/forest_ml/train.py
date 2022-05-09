import pathlib
from .parse_dataset import get_dataset
from .feature_engineering import feature_engineering
from .k_fold_cross_validation import k_fold_cross_validation
from .nested_cross_validation import nested_cross_validation

import mlflow
import mlflow.sklearn
import click


@click.command()
@click.option(
    "--path-to-dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    show_default=True,
    help="A path to the file with your dataset",
)
@click.option(
    "--path-save-model",
    default="data/model.joblib",
    required=False,
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    show_default=True,
    help="A path where to save the trained model",
)
@click.option(
    "--random-state",
    default=42,
    type=click.IntRange(0, 1000000000),
    show_default=True,
    required=False,
    help="random_state for train test split, model training",
)
@click.option(
    "--test-split-ratio",
    default=0.3,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
    required=False,
    help="Test data ratio, 0.3 by default",
)
@click.option(
    "--use-scaler",
    default=False,
    type=bool,
    show_default=True,
    required=False,
    help="Whether to use a scaler on data or not, False by default",
)
@click.option(
    "--cross-validation-type",
    type=click.Choice(["nested", "k-fold"]),
    default="nested",
    show_default=True,
    help="What type of cross-validation to use",
    required=False,
)
@click.option(
    "--model",
    type=click.Choice(
        [
            "RandomForestClassifier",
            "LogisticRegression",
            "KNeighborsClassifier",
            "ExtraTreesClassifier",
        ]
    ),
    default="RandomForestClassifier",
    show_default=True,
    help="What machine learning model to use",
    required=False,
)
@click.option(
    "--fetengtech",
    type=click.Choice(
        [
            "1",
            "2",
        ]
    ),
    default="1",
    show_default=True,
    help="What feature engineering technique to use",
    required=False,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
    required=False,
    help="hyperparameter for random forest and extratreeclassifier",
)
@click.option(
    "--n-estimators",
    default=None,
    type=int,
    show_default=True,
    required=False,
    help="hyperparameter for random forest and extratreeclassifier",
)
@click.option(
    "--max-features",
    default=None,
    type=str,
    show_default=True,
    required=False,
    help="hyperparameter for random forest and extratreeclassifier",
)
@click.option(
    "--n-neighbors",
    default=None,
    type=int,
    show_default=True,
    required=False,
    help="hyperparameter for knn",
)
@click.option(
    "--weights",
    default=None,
    type=str,
    show_default=True,
    required=False,
    help="hyperparameter for knn",
)
@click.option(
    "--max-iter",
    default=None,
    type=int,
    show_default=True,
    required=False,
    help="hyperparameter for logistic regression",
)
@click.option(
    "--C",
    default=None,
    type=float,
    show_default=True,
    required=False,
    help="hyperparameter for logistic regression",
)
@click.option(
    "--penalty",
    default=None,
    type=str,
    show_default=True,
    required=False,
    help="hyperparameter for logistic regression",
)
@click.option(
    "--solver",
    default=None,
    type=str,
    show_default=True,
    required=False,
    help="hyperparameter for logistic regression",
)
def train(
    path_to_dataset: pathlib.Path,
    path_save_model: pathlib.Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    cross_validation_type: str,
    model: str,
    fetengtech: str,
    max_depth: int,
    n_estimators: int,
    max_features: str,
    n_neighbors: int,
    weights: str,
    max_iter: int,
    c: float,
    penalty: str,
    solver: str,
) -> None:
    params_list = {
        "path_to_dataset": path_to_dataset,
        "path_save_model": path_save_model,
        "random_state": random_state,
        "test_split_ratio": test_split_ratio,
        "use_scaler": use_scaler,
        "cross_validation_type": cross_validation_type,
        "model": model,
        "fetengtech": fetengtech,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "n_neighbors": n_neighbors,
        "weights": weights,
        "max_iter": max_iter,
        "C": c,
        "penalty": penalty,
        "solver": solver,
    }
    params = dict(filter(lambda x: x[1] is not None, params_list.items()))
    print("params", params)

    X, y = get_dataset(
        csv_path=params["path_to_dataset"],
        split_into_train_test=False,
        random_state=params["random_state"],
        test_split_ratio=params["test_split_ratio"],
    )
    X = feature_engineering(dataset=X, feature_eng_tech=params["fetengtech"])
    with mlflow.start_run():
        if params["cross_validation_type"] == "nested":
            nested_cross_validation(X, y, params)
        ###
        else:
            k_fold_cross_validation(X, y, params)
