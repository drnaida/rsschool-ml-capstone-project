import argparse
import pathlib
from .parse_dataset import get_dataset
from joblib import dump
from .model_pipeline import create_pipeline
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--path-to-dataset', type=pathlib.Path, required=True, help='A path to the file with your dataset')
parser.add_argument('--path-save-model', type=pathlib.Path, required=False, default='models/model.joblib', help='A path where to save the trained model')
parser.add_argument('--random-state', type=int, required=False, default=42, help='random_state for train test split, model training and etc., must be an integer')
parser.add_argument('--test-split-ratio', type=int, choices=range(0, 1), required=False, default=0.3, help='Test data ratio, 0.3 by default')
parser.add_argument('--use-scaler', type=bool, required=False, default=False, help='Whether to use a scaler on data or not, False by default')
parser.add_argument('--max-iter', type=int, required=False, default=100, help='What max iter to use')
parser.add_argument('--logreg-c', type=float, required=False, default=1.0, help='What inverse of regularization strength to have (for logistical regression)')
arguments = parser.parse_args()

def train(
        path_to_dataset: pathlib.Path=arguments.path_to_dataset,
        path_save_model: pathlib.Path=arguments.path_save_model,
        random_state: int=arguments.random_state,
        test_split_ratio: float=arguments.test_split_ratio,
        use_scaler: bool=arguments.use_scaler,
        max_iter: int=arguments.max_iter,
        logreg_c: float=arguments.logreg_c,
) -> None:
    X, y = get_dataset(
        csv_path=path_to_dataset, split_into_train_test=False, random_state=random_state, test_split_ratio=test_split_ratio
    )
    pipeline = create_pipeline(use_scaler=use_scaler)
    accuracy = cross_val_score(pipeline, X, y)
    print(accuracy)
    dump(pipeline, path_save_model)
