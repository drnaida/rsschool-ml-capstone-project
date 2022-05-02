from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def create_pipeline(
    use_scaler: bool=False, max_iter: int=100, logreg_C: float=1.0, random_state: int=42
) -> Pipeline:
    pipeline_steps = []
    clf = RandomForestClassifier
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            clf(
                random_state=random_state
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)