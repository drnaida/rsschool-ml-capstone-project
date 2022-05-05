def feature_engineering(dataset, feature_engineering_tech):
    if feature_engineering_tech == '1':
        dataset['Euclidian_Distance_To_Hydrology'] = \
            (dataset['Horizontal_Distance_To_Hydrology'] ** 2 + \
            dataset['Vertical_Distance_To_Hydrology']**2) ** 0.5
        print(dataset)
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

        dataset["Soil_Type_sum"] = 0
        for soil_num in range(1, 41):
            column_name = "Soil_Type" + str(soil_num)
            dataset["Soil_Type_sum"] += 2 ** dataset[column_name]
            dataset.drop(column_name, inplace=True, axis=1)
        print(dataset)
        return dataset