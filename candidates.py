from numpy import isin
import pandas as pd

import torch

if torch.cuda.is_available():
    import cudf  # type: ignore

    pd_or_cudf = cudf
else:
    pd_or_cudf = pd


def create_candidates(
    transactions_df, customers_df, articles_df, article_pairs, **kwargs
):
    """
    will return candidates in form customer_id_article_id
    will also return 'features' dict, where:
        - format is {<column_name>:<features_df},
        - column_name is a column, or tuple of columns, in articles, customers or transactions
        - features_df.index values correspond to values of column_name column
        - columns of features_df are engineered features unique to the df.index values
    """
    features = {}
    last_week_number = transactions_df["week_number"].max()

    #######################################################
    # past x weeks customer purchases (counts from always)
    #####################################################
    customer_article_df = (
        transactions_df.groupby(["customer_id", "article_id"])
        .agg(
            {
                "week_number": "max",
                "t_dat": "max",
                "price": "count",
            }
        )
        .rename(
            columns={
                "week_number": "ca_last_purchase_week",
                "t_dat": "ca_last_purchase_date",
                "price": "ca_purchase_count",
            }
        )
        .sort_values("ca_purchase_count", ascending=False)
    )

    features["customer_article"] = (["customer_id", "article_id"], customer_article_df)
    customer_article_cand = (
        customer_article_df.query(
            f"ca_last_purchase_week >= {last_week_number - kwargs['ca_num_weeks'] + 1}"
        )
        .reset_index()[["customer_id", "article_id"]]
        .drop_duplicates()
    )

    ###########################################################
    # customer_last_weeks purchases (and their pairs)
    ###########################################################

    clw_df = transactions_df[["customer_id", "article_id", "t_dat"]].copy()
    clw_df["t_dat"] = pd_or_cudf.to_datetime(clw_df["t_dat"])

    last_customer_purchase_dat = clw_df.groupby("customer_id")["t_dat"].max()
    clw_df["max_cust_dat"] = clw_df["customer_id"].map(last_customer_purchase_dat)
    clw_df["diff_cust_dat"] = (clw_df["max_cust_dat"] - clw_df["t_dat"]).dt.days
    clw_df = clw_df.query(f"diff_cust_dat <= {kwargs['clw_num_weeks'] * 7 - 1}").copy()

    clw_df = (
        clw_df.groupby(["customer_id", "article_id"])["t_dat"].count().reset_index()
    )
    clw_df.columns = [
        "customer_id",
        "article_id",
        "clw_count",
    ]

    del last_customer_purchase_dat

    clw_pairs_df = clw_df.copy()
    clw_pairs_df["article_id"] = clw_pairs_df["article_id"].map(article_pairs)
    clw_pairs_df = (
        clw_pairs_df.groupby(["customer_id", "article_id"])["clw_count"]
        .max()
        .reset_index()
    )
    clw_pairs_df.columns = ["customer_id", "article_id", "clw_pair_count"]

    cust_last_week_cand = clw_df[["customer_id", "article_id"]].drop_duplicates()
    cust_last_week_pair_cand = clw_pairs_df[
        ["customer_id", "article_id"]
    ].drop_duplicates()

    clw_df = clw_df.set_index(["customer_id", "article_id"])[["clw_count"]].copy()
    features["customer_last_week"] = (["customer_id", "article_id"], clw_df)

    clw_pairs_df = clw_pairs_df.set_index(["customer_id", "article_id"])[
        ["clw_pair_count"]
    ].copy()
    features["customer_last_week_pairs"] = (["customer_id", "article_id"], clw_pairs_df)

    #############################
    # recently popular items
    #############################

    article_purchases_df = (
        transactions_df.query(
            f"week_number >= {last_week_number - kwargs['pa_num_weeks'] + 1}"
        )
        .groupby("article_id")
        .agg(
            {
                "week_number": "max",
                "t_dat": "count",
            }
        )
        .rename(
            columns={
                "week_number": "pa_last_purchase_week",
                "t_dat": "pa_purchase_count",
            }
        )
        .sort_values("pa_purchase_count", ascending=False)
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

    features["popular_articles"] = (["article_id"], article_purchases_df)

    candidates = (
        pd_or_cudf.concat(
            [
                customer_article_cand,
                cust_last_week_cand,
                cust_last_week_pair_cand,
                popular_items_cand,
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return candidates, features


def add_features_to_candidates(candidates_df, features, customers_df, articles_df):
    """
    adds fields needed to merge in features
    and merges features in
    """
    for features_key in features:
        col_names, feature_df = features[features_key]

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


def filter_candidates(candidates, transactions_df, **kwargs):
    recent_art_weeks = kwargs.get("filter_recent_art_weeks", None)
    if recent_art_weeks:
        recent_articles = transactions_df.query(
            f"week_number >= {kwargs['label_week'] - recent_art_weeks}"
        )["article_id"].drop_duplicates()
        candidates = candidates[candidates["article_id"].isin(recent_articles)].copy()

    return candidates
