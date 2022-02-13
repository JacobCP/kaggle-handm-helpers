import os
import pandas as pd

competition_directory = "/kaggle/input/h-and-m-personalized-fashion-recommendations/"
file_names = [
    "articles.csv",
    "customers.csv",
    "sample_submission.csv",
    "transactions_train.csv",
]


def load_data(data_path=competition_directory, files=file_names):
    dtypes = {"article_id": str}
    loaded_dfs = []
    for file_name in file_names:
        file_path = os.path.join(data_path, file_name)
        loaded_dfs.append(pd.read_csv(file_path), dtype=dtypes)

    return loaded_dfs
