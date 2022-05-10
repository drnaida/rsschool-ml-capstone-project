from sklearn.decomposition import TruncatedSVD
import pandas as pd
from typing import Any


def feature_engineering(dataset: pd.DataFrame, feature_eng_tech: str) -> Any:
    if feature_eng_tech == "1":
        dataset["Euclidian_Distance_To_Hydrology"] = (
            dataset["Horizontal_Distance_To_Hydrology"] ** 2
            + dataset["Vertical_Distance_To_Hydrology"] ** 2
        ) ** 0.5
        print(type(dataset.to_numpy()))
        return dataset.to_numpy()
    elif feature_eng_tech == "2":
        columns = [
            "Elevation",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
        for column in columns:
            for degree in range(2, 5):
                dataset[column + "_" + str(degree)] = dataset[column] ** degree
        truncatedSVD = TruncatedSVD(n_components=35)
        reduced_dimensiallity_dataset = truncatedSVD.fit_transform(dataset)
        print(type(reduced_dimensiallity_dataset))
        return reduced_dimensiallity_dataset
