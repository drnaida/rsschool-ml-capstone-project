import argparse
import pathlib
from .parse_dataset import get_dataset
from joblib import dump
from .model_pipeline import create_pipeline
from sklearn.model_selection import cross_validate

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--path-to-dataset', type=pathlib.Path, required=True, help='A path to the file with your dataset')
parser.add_argument('--path-save-model', type=pathlib.Path, required=False, default='models/model.joblib', help='A path where to save the trained model')
parser.add_argument('--random-state', type=int, required=False, default=42, help='random_state for train test split, model training and etc., must be an integer')
parser.add_argument('--test-split-ratio', type=int, choices=range(0, 1), required=False, default=0.3, help='Test data ratio, 0.3 by default')
parser.add_argument('--use-scaler', type=bool, required=False, default=False, help='Whether to use a scaler on data or not, False by default')
parser.add_argument('--model', type=str, required=False, choices=['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'ExtraTreesClassifier'], default='RandomForestClassifier', help='What machine learning model to use')

#hyperparameters for random forest classifier
parser.add_argument('--max-depth', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--min-sample-split', type=int, required=False, default=2, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-leaf-nodes', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--min-samples-leaf', type=int, required=False, default=1, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--n-estimators', type=int, required=False, default=100, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-samples', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-features', type=int, required=False, default='auto', help='hyperparameter for random forest and extratreeclassifier')

#knn
parser.add_argument('--n-neighbors', type=int, required=False, default=5, help='hyperparameter for knn')
parser.add_argument('--weights', type=str, required=False, default='uniform', help='hyperparameter for knn')
parser.add_argument('--leaf_size', type=int, required=False, default=30, help='hyperparameter for knn')

#logistic regression
parser.add_argument('--max-iter', type=int, required=False, default=100, help='hyperparameter for logistic regression')
parser.add_argument('--logreg-c', type=float, required=False, default=1.0, help='hyperparameter for logistic regression')
parser.add_argument('--penalty', type=str, required=False, default='l2', help='hyperparameter for logistic regression')
parser.add_argument('--solver', type=str, required=False, default='lbfgs', help='hyperparameter for logistic regression')
arguments = parser.parse_args()

def train(
        path_to_dataset: pathlib.Path=arguments.path_to_dataset,
        path_save_model: pathlib.Path=arguments.path_save_model,
        random_state: int=arguments.random_state,
        test_split_ratio: float=arguments.test_split_ratio,
        use_scaler: bool=arguments.use_scaler,
        model: str=arguments.model,
        max_depth: int=arguments.max_depth,
        min_sample_split: int=arguments.min_sample_split,
        max_leaf_nodes: int=arguments.max_leaf_nodes,
        min_samples_leaf: int=arguments.min_samples_leaf,
        n_estimators: int=arguments.n_estimators,
        max_samples: int=arguments.max_samples,
        max_features: int=arguments.max_features,
        n_neighbors: int=arguments.n_neighbors,
        weights: int=arguments.weights,
        leaft_size: int=arguments.leaf_size,
        max_iter: int=arguments.max_iter,
        logreg_c: float=arguments.logreg_c,
        penalty: str=arguments.penalty,
        solver: str=arguments.solver
) -> None:
    X, y = get_dataset(
        csv_path=path_to_dataset, split_into_train_test=False, random_state=random_state, test_split_ratio=test_split_ratio
    )
    pipeline = create_pipeline(model=model, use_scaler=use_scaler)
    scoring = {'acc': 'accuracy',
               'f1_weighted': 'f1_weighted',
               'roc_auc_ovr': 'roc_auc_ovr'}
    scores = cross_validate(pipeline, X, y, scoring=scoring)
    print(scores)
    dump(pipeline, path_save_model)
