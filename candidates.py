from numpy import isin
import pandas as pd

import torch

if torch.cuda.is_available():
    import cudf  # type: ignore

    pd_or_cudf = cudf
else:
    pd_or_cudf = pd


def create_candidates(transactions_df, customers_df, articles_df, **kwargs):
    """
    will return candidates in form customer_id_article_id
    will also return 'features' dict, where:
        - format is {<column_name>:<features_df},
        - column_name is a column, or tuple of columns, in articles, customers or transactions
        - features_df.index values correspond to values of column_name column
        - columns of features_df are engineered features unique to the df.index values
    """
    features = {}

    # past customer/article purchases
    past_purchases_df = (
        transactions_df.groupby(["customer_id", "article_id"])
        .agg(
            {
                "week_number": "max",
                "t_dat": "count",
            }
        )
        .rename(
            columns={
                "week_number": "last_cust_art_purchase_week",
                "t_dat": "cust_art_purchase_count",
            }
        )
        .sort_values("cust_art_purchase_count", ascending=False)
    )

    features[("customer_id", "article_id")] = past_purchases_df
    past_purchases_cand = past_purchases_df.reset_index()[
        ["customer_id", "article_id"]
    ].drop_duplicates()

    # recently popular items
    article_purchases_df = (
        transactions_df.groupby("article_id")
        .agg(
            {
                "week_number": "max",
                "t_dat": "count",
            }
        )
        .rename(
            columns={
                "week_number": "last_art_purchase_week",
                "t_dat": "art_purchase_count",
            }
        )
        .sort_values("art_purchase_count", ascending=False)
    )

    num_recent_items = kwargs.get("num_recent_items", 12)
    recent_popular_items = article_purchases_df.index[:num_recent_items]
    popular_items_df = pd_or_cudf.DataFrame(
        {"article_id": pd_or_cudf.Series(recent_popular_items), "join_col": 1}
    )

    popular_items_cand = pd_or_cudf.DataFrame(
        {"customer_id": customers_df["customer_id"], "join_col": 1}
    )
    popular_items_cand = popular_items_cand.merge(popular_items_df, on="join_col")
    del popular_items_cand["join_col"]

    features["article_id"] = article_purchases_df

    candidates = (
        pd_or_cudf.concat([past_purchases_cand, popular_items_cand])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return candidates, features


def filter_candidates(candidates):
    """
    manual filtering logic of candidates, to keep it from getting out of hand
    """
    return candidates


def add_features_to_candidates(candidates_df, features, customers_df, articles_df):
    """
    adds fields needed to merge in features
    and merges features in
    """
    for features_key in features:
        feature_df = features[features_key]

        if isinstance(features_key, tuple):
            col_names = list(features_key)
        else:
            col_names = [features_key]

        # add the key to our df so we can merge the features in
        for col_name in col_names:
            if col_name not in candidates_df:
                if col_name in customers_df:
                    col_name_dict = customers_df.set_index("customer_id")[col_name]
                    candidates_df[col_name] = candidates_df["customer_id"].map(
                        col_name_dict
                    )
                elif col_name in articles_df:
                    col_name_dict = articles_df.set_index("article_id")[col_name]
                    candidates_df[col_name] = candidates_df["article_id"].map(
                        col_name_dict
                    )

        # now we can add the features
        candidates_df = candidates_df.merge(feature_df, how="left", on=col_names)

    return candidates_df