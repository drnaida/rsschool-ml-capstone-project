import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv('../..//data/train.csv')
profile = ProfileReport(df, title="Pandas Profiling Report Forest Dataset", explorative=True)
profile.to_file(output_file="../../eda/eda_forest_dataset.html")