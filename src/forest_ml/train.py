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
        help="random_state for train test split, model training and etc., must be an integer",
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
        "--min-sample-split",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for random forest and extratreeclassifier",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        required=False,
        default=None,
        help="hyperparameter for random forest and extratreeclassifier",
    )
    parser.add_argument(
        "--min-samples-leaf",
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
        "--max-samples",
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
    parser.add_argument(
        "--leaf_size",
        type=int,
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
    X = feature_engineering(dataset=X, feature_engineering_tech=params["fetengtech"])
    with mlflow.start_run():
        pipeline = create_pipeline(**params)
        scoring = {
            "acc": "accuracy",
            "f1_weighted": "f1_weighted",
            "roc_auc_ovr": "roc_auc_ovr",
        }
        scores = cross_validate(pipeline, X, y, scoring=scoring)
        avg_accuracy = np.mean(scores["test_acc"])
        avg_f1 = np.mean(scores["test_f1_weighted"])
        avg_roc_auc_ovr = np.mean(scores["test_roc_auc_ovr"])
        dump(pipeline, params["path_save_model"])
        mlflow.sklearn.log_model(
            sk_model=pipeline, artifact_path=os.path.dirname(params["path_save_model"])
        )
        mlflow.log_param("model_type", params["model"])
        mlflow.log_param("feat_eng_type", params["fetengtech"])
        params = _params_for_models(params)
        for param in params:
            mlflow.log_param(param, params[param])
        mlflow.log_metric("accuracy", avg_accuracy)
        mlflow.log_metric("f1_weighted", avg_f1)
        mlflow.log_metric("roc_auc_ovr", avg_roc_auc_ovr)
