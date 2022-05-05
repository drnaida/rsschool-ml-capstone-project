from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def create_pipeline(
    model: str = RandomForestClassifier, use_scaler: bool=False, **params
) -> Pipeline:
    pipeline_steps = []
    if model == 'RandomForestClassifier':
        clf = RandomForestClassifier(
                **params
            )
    elif model == 'ExtraTreesClassifier':
        clf = ExtraTreesClassifier(
                **params
            )
    elif model == 'KNeighborsClassifier':
        clf = KNeighborsClassifier(
                **params
            )
    else:
        clf = LogisticRegression(
                **params
            )
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            clf,
        )
    )
    return Pipeline(steps=pipeline_steps)