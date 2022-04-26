import cudf
import pandas as pd
import numpy as np


def get_group_lengths(df):
    """
    Get group_lengths to pass to LGBMRanker
    """
    df = df.to_pandas()

    return list(df.groupby("customer_id")["article_id"].count())


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


def create_predictions(ids_df, preds):
    ids_df["pred"] = preds
    ids_df = ids_df.sort_values(["customer_id", "pred"], ascending=False)
    ids_df = cudf_groupby_head(ids_df, "customer_id", 12)
    predictions = ids_df.groupby("customer_id")["article_id"].agg(list)

    return predictions


def pred_in_batches(model, features_df, batch_size=1000000):
    preds = []
    num_batches = int(len(features_df) / 1000000) + 1
    for batch in range(num_batches):
        batch_df = features_df.iloc[batch * batch_size : (batch + 1) * batch_size, :]
        preds.append(model.predict(batch_df))

    return np.concatenate(preds)
