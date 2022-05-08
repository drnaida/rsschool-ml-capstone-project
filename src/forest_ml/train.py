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
    X = feature_engineering(dataset=X, feature_eng_tech=params["fetengtech"])
    with mlflow.start_run():
        if params['cross_validation_type'] == 'nested':
            # configure the cross-validation procedure
            cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
            # enumerate splits
            best_models = []
            for train_ix, test_ix in cv_outer.split(X, y):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                y_train, y_test = y[train_ix], y[test_ix]
                # configure the cross-validation procedure
                cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
                # define the model
                params_new = {'model': params['model'], 'use_scaler': params['use_scaler']}
                model = create_pipeline(**params_new)
                # define search space
                space = dict()
                space_name = str(params['model']) + '__' + 'n_estimators'
                space[space_name] = [10, 100, 500]
                # define search
                search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
                scoring = {
                    "acc": "accuracy",
                    "f1_weighted": "f1_weighted",
                    "roc_auc_ovr": "roc_auc_ovr",
                }


                result = search.fit(X_train, y_train)
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)
                pred_prob = best_model.predict_proba(X_test)
                # evaluate the model
                acc = accuracy_score(y_test, yhat)
                roc_auc_ovr = roc_auc_score(y_test, pred_prob, multi_class='ovr')
                f1_avg = f1_score(y_test, yhat, average='weighted')
                # store the result
                best_models.append({'acc': acc, 'roc_auc_ovr': roc_auc_ovr, 'f1_avg': f1_avg, 'model': best_model})

            sorted_best_models = sorted(best_models, key=lambda d: d['acc'])
            best_model_after_nested = sorted_best_models[-1]

        ###
        else:
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
            artifact_path_for_model = os.path.dirname(params["path_save_model"])
            mlflow.sklearn.log_model(
                sk_model=pipeline, artifact_path=artifact_path_for_model
            )
            mlflow.log_param("model_type", params["model"])
            mlflow.log_param("feat_eng_type", params["fetengtech"])
            params = _params_for_models(params)
            for param in params:
                mlflow.log_param(param, params[param])
            mlflow.log_metric("accuracy", avg_accuracy)
            mlflow.log_metric("f1_weighted", avg_f1)
            mlflow.log_metric("roc_auc_ovr", avg_roc_auc_ovr)
