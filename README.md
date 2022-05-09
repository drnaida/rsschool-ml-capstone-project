# rsschool-ml-capstone-project

Screenshot for #8: 12 experiments recorded by mlflow using
3 hyperparameters sets, 2 feature engineering techniques,
and 2 machine learning models.

![](mlflow-results-with-diff-models-hyperparameters-feature-engineering.PNG)

Information for #7: In the commit named (73f16a3bab7c4a2a9c3ef29d4deb62c3f5e6fa26) or (Feat: run 12 experiments with diff models, hyperparameters and feature engineering and track tham with mlflow
) you can see that I implemented k fold cross validation using cross_validate().

# Usage

1. Clone this repository
```
git clone link-to-the-reposotory
```

2. Download [Forest Cover dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction) to data/ folder
To underline, if you download this dataset to another folder rather than data/, don't forget to use a path in --path-to-dataset respectivelly.

3. Make sure Python 3.9 and Poetry 1.1.11 are installed on your machine

4. Install the project dependepcies from the root of the cloned repository

```
poetry install --no-dev
```

# Use train.py (train the model)

Run train (the path below change to the path in which you have train.csv file with the dataset)

```
poetry run train --path-to-dataset D:\dev\rsschool-ml-capstone\data\train.csv
```

Train script has several arguments, but only --path-to-dataset is required.

I'm going to specify other arguments here:

"--path-save-model" path where to save the model with the name of the file, by default it is models/model.joblib

"--random-state" random_state for train test split, 42 by default

"--test-split-ratio" Test data ratio between 0 and 1

"--use-scaler" whether to use a scaler on the dataset, can be True or False

"--fetengtech" what feature engineering technique to use, can be 1 (in this case will be added a new column Eucledean distance to hydrology) or 2 (in this case columns

"Elevation", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points" will be multiplied to degrees from 2 to 4 and after that truncatedSVD will be used)

"--model" chooses the model, it can be "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "ExtraTreesClassifier" (RandomForestClassifier by default)

"--cross-validation-type" states whether to use nested cross-validation or k-fold (you type 'nested' or 'k-fold', 'nested' by default). **Important: if you choose nested than all of the hyperparameters that you specify in the CLI will NOT be used,
so if you want to specify your own hyperparameters choose --cross-validation-type 'k-fold'**

If you choose "--cross-validation-type 'k-fold'" than you may specify parameters below:
**To underline, use only hyperparameters that work for the model you chose in --model**
**To underline, if you don't specify a hyperparameter, a default value of hyperparameter from the scikit learn library will be used**

For RandomForestClassifier or ExtraTreesClassifier you can specify:

"--max-depth"

"--n-estimators"

"--max-features"

For KNeighborsClassifier you can specify:

"--n-neighbors"

"--weights"

For LogisticRegression you can specify:

"--max-iter"

"--C"

"--penalty"

"--solver"

**If you specify --cross-validation-type as 'nested', than GridSearchCV will iterate only through
hyperparameters that are named above (values through which it will iterate you can see in the code 'nested_cross_validation.py' file).**

## Examples of how you can run the train.py

With RandomForestClassifier and nested cross validation type
```
poetry run train --path-to-dataset D:\dev\rsschool-ml-capstone-project\data\train.csv --model RandomForestClassifier --fetengtech 1 --cross-validation-type 'nested'
```

With Logistic Regression and K-fold validation type:
```
poetry run train --path-to-dataset D:\dev\rsschool-ml-capstone-project\data\train.csv --model LogisticRegression --C 0.2 --max-iter 200 --penalty 'l2' --solver 'sag' --fetengtech 2 --cross-validation-type 'k-fold'
```

# Use eda.py (make an explaratory data analysis using pandas-profiling)

To run the eda.py, type in the console at the root of the project:
```
poetry run eda --path-to-dataset D:\dev\rsschool-ml-capstone-project\data\train.csv
```

The only argument that you must specify is --path-to-dataset.

# Development (without nox)

Please download the dev packages:

```
poetry install
```

**To underline, all of the commands must be executed at the root of the project**

## Using black

```
poetry add --dev black
```

**To format scripts in src (development scripts)**

```
poetry run black src
```

**To check whether code in src folder meets black requirements**

```
poetry run black --check src
```

**To format scripts in tests folder (tests scripts)**

```
poetry run black tests
```

**To check whether code in tests folder meets black requirements**

```
poetry run black --check tests
```

Black will automatically format your code.

## Using flake8

To underline, I have a .flake8 file in which I specify that my max length of line is 88 symbols
because I ran into conflicts between black and flake8 (flake8 would tell me that line is too long
but when I put arguments of the function that exceeded the line limit on the new line black
would tell me that it is not good)

**Download flake8**

```
poetry add --dev flake8
```

**Use flake8 / see what errors you have**

```
poetry run flake8
```

After that you will have to fix the mistakes with your hands. You will see an empty output
if there is nothing to fix.

## Using mypy

```
poetry add --dev mypy
```

# Development WITH nox

Please download the dev packages if you haven't done it before:

```
poetry install
```


