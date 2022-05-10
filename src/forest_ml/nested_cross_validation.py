import os
from joblib import dump
from .model_pipeline import create_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from typing import Any
import pandas as pd

import mlflow
import mlflow.sklearn

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def nested_cross_validation(X: Any, y: pd.Series, params: dict[str, Any]) -> None:
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    # enumerate splits
    best_models: list[dict[str, Any]] = []
    space: dict[str, Any] = {}
    params_for_grid_search = [
        "n_estimators",
        "max_depth",
        "min_sample_split",
        "max_features",
        "n_neighbors",
        "weights",
        "max_iter",
        "C",
        "penalty",
        "solver",
    ]
    for param in params_for_grid_search:
        space_name = str(params["model"]) + "__" + param
        if (params["model"] == "RandomForestClassifier") or (
            params["model"] == "ExtraTreesClassifier"
        ):
            # random forest
            # if param == "n_estimators":
            #     space[space_name] = [10, 100, 300]
            # if param == "max_depth":
            #     space[space_name] = [3, 10, 20]
            # if param == "max_features":
            #     space[space_name] = ["log2", "sqrt"]
            # random forest
            if param == "n_estimators":
                space[space_name] = [10]
            if param == "max_depth":
                space[space_name] = [3]
            if param == "max_features":
                space[space_name] = ["log2"]
        if params["model"] == "KNeighborsClassifier":
            if param == "n_neighbors":
                space[space_name] = [3, 5, 7]
            if param == "weights":
                space[space_name] = ["uniform", "distance"]
        # logistic regression
        if params["model"] == "LogisticRegression":
            if param == "max_iter":
                space[space_name] = [100, 150, 200]
            if param == "C":
                space[space_name] = [0.01, 0.1, 1]
            if param == "penalty":
                space[space_name] = ["l2", "l1"]
            if param == "solver":
                space[space_name] = ["liblinear", "saga", "sag"]
    print("space", space)
    for train_ix, test_ix in cv_outer.split(X, y):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        # define the model
        params_new = {"model": params["model"], "use_scaler": params["use_scaler"]}
        model = create_pipeline(**params_new)
        # define search
        search = GridSearchCV(model, space, scoring="accuracy", cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        best_params = result.best_params_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        pred_prob = best_model.predict_proba(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        roc_auc_ovr = roc_auc_score(y_test, pred_prob, multi_class="ovr")
        f1_avg = f1_score(y_test, yhat, average="weighted")
        # store the result
        best_models.append(
            {
                "acc": acc,
                "roc_auc_ovr": roc_auc_ovr,
                "f1_avg": f1_avg,
                "model": best_model,
                "params": best_params,
            }
        )

    sorted_best_models = sorted(best_models, key=lambda d: d["acc"])  # type: ignore
    best_model_after_nested = sorted_best_models[-1]
    final_model = best_model_after_nested["model"]
    print(final_model)
    final_model.fit(X, y)
    dump(final_model, params["path_save_model"])
    # replace accuracies and save to mlflow
    mean_acc = sum(d["acc"] for d in sorted_best_models) / len(sorted_best_models)
    mean_roc_auc_ovr = sum(d["roc_auc_ovr"] for d in sorted_best_models) / len(
        sorted_best_models
    )
    mean_f1_avg = sum(d["f1_avg"] for d in sorted_best_models) / len(sorted_best_models)
    # save to mlflow the model
    artifact_path_for_model = os.path.dirname(params["path_save_model"])
    mlflow.sklearn.log_model(
        sk_model=final_model, artifact_path=artifact_path_for_model
    )
    mlflow.log_param("model_type", params["model"])
    mlflow.log_param("feat_eng_type", params["fetengtech"])
    # save params
    grid_search_parameters = best_model_after_nested["params"]
    for param in grid_search_parameters:
        what_to_replace = str(params["model"]) + "__"
        new_param_name = param.replace(what_to_replace, "")
        print("-========")
        print(new_param_name, grid_search_parameters[param])
        mlflow.log_param(new_param_name, grid_search_parameters[param])
    # save metrics
    mlflow.log_metric("accuracy", mean_acc)
    mlflow.log_metric("f1_weighted", mean_roc_auc_ovr)
    mlflow.log_metric("roc_auc_ovr", mean_f1_avg)

    ### some tests
    features = X
    target = y
    print(params["path_save_model"])
    loaded_model = pickle.load(open(params["path_save_model"], "rb"))
    accuracy = cross_val_score(loaded_model, features, target, scoring="accuracy", cv=5)
    avg_accuracy = np.mean(accuracy)
    assert result.exit_code == 0
    assert 0 < avg_accuracy < 1
