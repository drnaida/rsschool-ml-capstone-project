from sklearn.decomposition import TruncatedSVD
def feature_engineering(dataset, feature_engineering_tech):
    if feature_engineering_tech == '1':
        dataset['Euclidian_Distance_To_Hydrology'] = \
            (dataset['Horizontal_Distance_To_Hydrology'] ** 2 + \
            dataset['Vertical_Distance_To_Hydrology']**2) ** 0.5
        return dataset
    elif feature_engineering_tech == '2':
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
        dataset_after_dimensiality_reduction = truncatedSVD.fit_transform(dataset)
        return dataset_after_dimensiality_reduction