import argparse
import pathlib
from .parse_dataset import get_dataset
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('--path-to-dataset', type=pathlib.Path, required=True,help='A path to the file with your dataset')
parser.add_argument('--path-save-model', type=pathlib.Path, required=False, default='../../models/', help='A path where to save the trained model')
parser.add_argument('--random-state', type=int, required=False, default=42, help='random_state for train test split, model training and etc., must be an integer')
parser.add_argument('--test-split-ratio', type=int, choices=range(0, 1), required=False, default=0.3, help='Test data ratio, 0.3 by default')
parser.add_argument('--use-scaler', type=bool, required=False, default=False, help='Whether to use a scaler on data or not, False by default')
parser.add_argument('--max-iter', type=bool, required=False, default=False, help='Whether to use a scaler on data or not, False by default')

def train():
    arguments = parser.parse_args()
    X_train, X_test, y_train, y_test = get_dataset(
        arguments.path_to_dataset,
        arguments.random_state,
        arguments.test_split_ratio,
    )
    print(X_train)
