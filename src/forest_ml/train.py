import argparse
import pathlib
import numpy as np
import os
from .parse_dataset import get_dataset
from joblib import dump
from .model_pipeline import create_pipeline
from .model_pipeline import _params_for_models
from .feature_engineering import feature_engineering
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from .k_fold_cross_validation import k_fold_cross_validation
from .nested_cross_validation import nested_cross_validation

import mlflow
import mlflow.sklearn


def get_params() -> dict:
    # arguments needed for model in general
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--path-to-dataset",
        type=pathlib.Path,
        required=True,
        help="A path to the file with your dataset",
    )
    parser.add_argument(
        "--path-save-model",
        type=pathlib.Path,
        required=False,
        default="models/model.joblib",
        help="A path where to save the trained model",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        required=False,
        default=42,
        help="random_state for train test split, model training",
    )
    parser.add_argument(
        "--test-split-ratio",
        type=int,
        choices=range(0, 1),
        required=False,
        default=0.3,
        help="Test data ratio, 0.3 by default",
    )
    parser.add_argument(
        "--use-scaler",
        type=bool,
        required=False,
        default=False,
        help="Whether to use a scaler on data or not, False by default",
    )
    parser.add_argument(
        "--cross-validation-type",
        type=str,
        required=False,
        default='nested',
        choices=[
            'nested',
            'k-fold'
        ],
        help="What type of cross-validation to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=[
            "RandomForestClassifier",
            "LogisticRegression",
            "KNeighborsClassifier",
            "ExtraTreesClassifier",
        ],
        default="RandomForestClassifier",
        help="What machine learning model to use",
    )
    parser.add_argument(
        "--fetengtech",
        type=str,
        required=False,
        choices=["1", "2"],
        default="1",
        help="What feature engineering technique to use",
    )

    # hyperparameters for random forest classifier
    parser.add_argument(
        "--max-depth",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for random forest and extratreeclassifier",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for random forest and extratreeclassifier",
    )
    parser.add_argument(
        "--max-features",
        type=str,
        required=False,
        default=None,
        help="hyperparameter for random forest and extratreeclassifier",
    )

    # knn
    parser.add_argument(
        "--n-neighbors",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for knn",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        default=None,
        help="hyperparameter for knn",
    )

    # logistic regression
    parser.add_argument(
        "--max-iter",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for logistic regression",
    )
    parser.add_argument(
        "--C",
        type=float,
        required=False,
        default=None,
        help="hyperparameter for logistic regression",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        required=False,
        default=None,
        help="hyperparameter for logistic regression",
    )
    parser.add_argument(
        "--solver",
        type=str,
        required=False,
        default=None,
        help="hyperparameter for logistic regression",
    )
    arguments = parser.parse_args()
    return arguments.__dict__


def train() -> None:
    params = dict(filter(lambda x: x[1] is not None, get_params().items()))
    print("params", params)

    X, y = get_dataset(
        csv_path=params["path_to_dataset"],
        split_into_train_test=False,
        random_state=params["random_state"],
        test_split_ratio=params["test_split_ratio"],
    )
    X = feature_engineering(dataset=X, feature_eng_tech=params["fetengtech"])
    with mlflow.start_run():
        if params['cross_validation_type'] == 'nested':
            nested_cross_validation(X, y, params)
        ###
        else:
            k_fold_cross_validation(X, y, params)
