import argparse
import pathlib
from .parse_dataset import get_dataset
from joblib import dump
from .model_pipeline import create_pipeline
from .feature_engineering import feature_engineering
from sklearn.model_selection import cross_validate

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

#arguments needed for model in general
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--path-to-dataset', type=pathlib.Path, required=True, help='A path to the file with your dataset')
parser.add_argument('--path-save-model', type=pathlib.Path, required=False, default='models/model.joblib', help='A path where to save the trained model')
parser.add_argument('--random-state', type=int, required=False, default=42, help='random_state for train test split, model training and etc., must be an integer')
parser.add_argument('--test-split-ratio', type=int, choices=range(0, 1), required=False, default=0.3, help='Test data ratio, 0.3 by default')
parser.add_argument('--use-scaler', type=bool, required=False, default=False, help='Whether to use a scaler on data or not, False by default')
parser.add_argument('--model', type=str, required=False, choices=['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'ExtraTreesClassifier'], default='RandomForestClassifier', help='What machine learning model to use')
parser.add_argument('--fetengtech', type=str, required=False, choices=['1', '2'], default='1', help='What feature engineering technique to use')

#hyperparameters for random forest classifier
parser.add_argument('--max-depth', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--min-sample-split', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-leaf-nodes', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--min-samples-leaf', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--n-estimators', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-samples', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')
parser.add_argument('--max-features', type=int, required=False, default=None, help='hyperparameter for random forest and extratreeclassifier')

#knn
parser.add_argument('--n-neighbors', type=int, required=False, default=None, help='hyperparameter for knn')
parser.add_argument('--weights', type=str, required=False, default=None, help='hyperparameter for knn')
parser.add_argument('--leaf_size', type=int, required=False, default=None, help='hyperparameter for knn')

#logistic regression
parser.add_argument('--max-iter', type=int, required=False, default=None, help='hyperparameter for logistic regression')
parser.add_argument('--C', type=float, required=False, default=None, help='hyperparameter for logistic regression')
parser.add_argument('--penalty', type=str, required=False, default=None, help='hyperparameter for logistic regression')
parser.add_argument('--solver', type=str, required=False, default=None, help='hyperparameter for logistic regression')
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
        leaf_size: int=arguments.leaf_size,
        max_iter: int=arguments.max_iter,
        C: float=arguments.C,
        penalty: str=arguments.penalty,
        solver: str=arguments.solver,
        fetengtech: str=arguments.fetengtech
) -> None:
    params = {
        'random_state': random_state,
        'max_depth': max_depth,
        'min_sample_split': min_sample_split,
        'max_leaf_nodes': max_leaf_nodes,
        'min_samples_leaf': min_samples_leaf,
        'n_estimators': n_estimators,
        'max_samples': max_samples,
        'max_features': max_features,
        'n_neighbors': n_neighbors,
        'weights': weights,
        'leaf_size': leaf_size,
        'max_iter': max_iter,
        'C': C,
        'penalty': penalty,
        'solver': solver
    }
    for key, value in dict(params).items():
        if value is None:
            del params[key]

    print('params', params)
    X, y = get_dataset(
        csv_path=path_to_dataset, split_into_train_test=False, random_state=random_state, test_split_ratio=test_split_ratio
    )
    X = feature_engineering(dataset=X, feature_engineering_tech=fetengtech)
    pipeline = create_pipeline(model=model, use_scaler=use_scaler, **params)
    scoring = {'acc': 'accuracy',
               'f1_weighted': 'f1_weighted',
               'roc_auc_ovr': 'roc_auc_ovr'}
    scores = cross_validate(pipeline, X, y, scoring=scoring)
    print(scores)
    dump(pipeline, path_save_model)
