from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def create_pipeline(
    model: str = RandomForestClassifier, use_scaler: bool=False, max_iter: int=100, logreg_C: float=1.0, random_state: int=42
) -> Pipeline:
    pipeline_steps = []
    if model == 'RandomForestClassifier':
        clf = RandomForestClassifier(
                random_state=random_state
            )
    elif model == 'ExtraTreesClassifier':
        clf = ExtraTreesClassifier(
                random_state=random_state
            )
    elif model == 'KNeighborsClassifier':
        clf = KNeighborsClassifier(
                random_state=random_state
            )
    else:
        clf = LogisticRegression(
                random_state=random_state
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