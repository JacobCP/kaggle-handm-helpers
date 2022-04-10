import os

import cudf
import pandas as pd

dtypes = {
    # articles
    "article_id": "int32",
    "prod_name": "category",
    "product_type_name": "category",
    "product_group_name": "category",
    "graphical_appearance_name": "category",
    "colour_group_name": "category",
    "perceived_colour_value_name": "category",
    "perceived_colour_master_name": "category",
    "department_name": "category",
    "garment_group_name": "category",
    "section_name": "category",
    "index_group_name": "category",
    "index_code": "category",
    "index_name": "category",
    "detail_desc": "category",
    "product_code": "int32",
    "product_type_no": "int16",
    "graphical_appearance_no": "int32",
    "colour_group_code": "int8",
    "perceived_colour_value_id": "int8",
    "perceived_colour_master_id": "int8",
    "department_no": "int16",
    "index_group_no": "int8",
    "section_no": "int8",
    "garment_group_no": "int16",
    # customers
    "FN": "category",
    "Active": "category",
    "club_member_status": "category",
    "fashion_news_frequency": "category",
    "age": "float32",
    "postal_code": "category",
    # transactions
    "price": "float32",
    "sales_channel_id": "int8",
}


competition_directory = "/kaggle/input/h-and-m-personalized-fashion-recommendations/"
file_names = [
    "articles.csv",
    "customers.csv",
    "sample_submission.csv",
    "transactions_train.csv",
]


def load_data(data_path=competition_directory, files=file_names):
    loaded_dfs = []
    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        df = cudf.read_csv(file_path)
        for column in df:
            if column in dtypes:
                df[column] = df[column].astype(dtypes[column])
        loaded_dfs.append(df)

    return loaded_dfs
