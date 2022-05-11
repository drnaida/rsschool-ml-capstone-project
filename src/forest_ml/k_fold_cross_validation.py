import numpy as np
import os
from joblib import dump
from .model_pipeline import create_pipeline
from .model_pipeline import _params_for_models
from sklearn.model_selection import cross_validate
import pandas as pd
from typing import Any

import mlflow
import mlflow.sklearn


def k_fold_cross_validation(X: Any, y: pd.Series, params: dict[str, Any]) -> None:
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
    if params["use_mlflow"]:
        artifact_path_for_model = os.path.dirname(params["path_save_model"])
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=artifact_path_for_model)
        mlflow.log_param("model_type", params["model"])
        mlflow.log_param("feat_eng_type", params["fetengtech"])
        params = _params_for_models(params)
        for param in params:
            mlflow.log_param(param, params[param])
        mlflow.log_metric("accuracy", avg_accuracy)
        mlflow.log_metric("f1_weighted", avg_f1)
        mlflow.log_metric("roc_auc_ovr", avg_roc_auc_ovr)
