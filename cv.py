import pandas as pd


def ground_truth(transactions_df):
    customer_trans = transactions_df[["customer_id", "article_id"]]
    customer_trans = customer_trans.groupby("customer_id")[["article_id"]].agg(set)
    customer_trans.columns = ["prediction"]
    customer_trans["prediction"] = customer_trans["prediction"].apply(list)
    customer_trans["prediction"] = customer_trans["prediction"].apply(
        lambda x: " ".join(x)
    )
    gt = customer_trans.reset_index()

    return gt
