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
    params_to_delete = [
        "path_to_dataset",
        "path_save_model",
        "test_split_ratio",
        "use_scaler",
        "model",
        "fetengtech",
        "cross_validation_type"
    ]

    for x in params_to_delete:
        if x in prepared.keys():
            del prepared[x]
    return prepared

def create_pipeline(**params) -> Pipeline:
    pipeline_steps = []
    if params["model"] == "RandomForestClassifier":
        clf = RandomForestClassifier(**_params_for_models(params=params))
    elif params["model"] == "ExtraTreesClassifier":
        clf = ExtraTreesClassifier(**_params_for_models(params=params))
    elif params["model"] == "KNeighborsClassifier":
        clf = KNeighborsClassifier(**_params_for_models(params=params))
    else:
        clf = LogisticRegression(**_params_for_models(params=params))
    if params["use_scaler"]:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            params["model"],
            clf,
        )
    )
    return Pipeline(steps=pipeline_steps)
