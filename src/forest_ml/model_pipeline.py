import copy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def _params_for_models(params: dict) -> dict:
    """
    # Deletes unnesecary params for models leaving only hyperparameters
    :param params: Original params from argparse
    :return: copy of original params without unnecessary args
    """
    prepared = copy.deepcopy(params)
    for x in [
        "path_to_dataset",
        "path_save_model",
        "test_split_ratio",
        "use_scaler",
        "model",
        "fetengtech",
    ]:
        if x in prepared.keys():
            del prepared[x]
    return prepared


def create_pipeline(**params) -> Pipeline:
    pipeline_steps = []
    if params["model"] == "RandomForestClassifier":
        clf = RandomForestClassifier(**_params_for_models(params))
    elif params["model"] == "ExtraTreesClassifier":
        clf = ExtraTreesClassifier(**_params_for_models(params))
    elif params["model"] == "KNeighborsClassifier":
        clf = KNeighborsClassifier(**_params_for_models(params))
    else:
        clf = LogisticRegression(**_params_for_models(params))
    if params["use_scaler"]:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            clf,
        )
    )
    return Pipeline(steps=pipeline_steps)
