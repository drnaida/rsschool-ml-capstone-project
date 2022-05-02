from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
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