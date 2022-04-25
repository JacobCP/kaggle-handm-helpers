import cudf
import pandas as pd


def create_recent_customer_candidates(transactions_df, recent_customer_weeks):
    last_week_number = transactions_df["week_number"].max()

    recent_customer_df = (
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

    features = (["customer_id", "article_id"], recent_customer_df)
    recent_customer_cand = (
        recent_customer_df.query(
            f"ca_last_purchase_week >= {last_week_number - recent_customer_weeks + 1}"
        )
        .reset_index()[["customer_id", "article_id"]]
        .drop_duplicates()
    )

    return recent_customer_cand, features


def create_last_customer_weeks_and_pairs(transactions_df, article_pairs_df, num_weeks):
    clw_df = transactions_df[["customer_id", "article_id", "t_dat"]].copy()

    # only transactions in "x" weeks before last customer purchase
    last_customer_purchase_dat = clw_df.groupby("customer_id")["t_dat"].max()
    clw_df["max_cust_dat"] = clw_df["customer_id"].map(last_customer_purchase_dat)
    clw_df["diff_cust_dat"] = clw_df["max_cust_dat"] - clw_df["t_dat"]
    clw_df = clw_df.query(f"diff_cust_dat <= {num_weeks * 7 - 1}").copy()

    # count purchased during that period
    clw_df = (
        clw_df.groupby(["customer_id", "article_id"])["t_dat"].count().reset_index()
    )
    clw_df.columns = [
        "customer_id",
        "article_id",
        "clw_count",
    ]

    del last_customer_purchase_dat

    # merge with pairs, and get max of:
    #  - sources' last week(s) purchase count
    #  - count and percent of customer pairs (see generating code for details)
    clw_pairs_df = clw_df.merge(article_pairs_df, on="article_id")
    clw_pairs_df = (
        clw_pairs_df.groupby(["customer_id", "pair_article_id"])[
            ["clw_count", "customer_count", "percent_customers"]
        ]
        .max()
        .reset_index()
    )
    clw_pairs_df.columns = [
        "customer_id",
        "article_id",
        "pair_clw_count",
        "pair_customer_count",
        "pair_percent_customers",
    ]
    clw_pairs_df = clw_pairs_df.query("pair_customer_count > 2").copy()

    cust_last_week_cand = clw_df[["customer_id", "article_id"]].drop_duplicates()
    cust_last_week_pair_cand = clw_pairs_df[
        ["customer_id", "article_id"]
    ].drop_duplicates()

    clw_df = clw_df.set_index(["customer_id", "article_id"])[["clw_count"]].copy()
    features = (["customer_id", "article_id"], clw_df)

    clw_pairs_df = clw_pairs_df.set_index(["customer_id", "article_id"])[
        ["pair_clw_count", "pair_customer_count", "pair_percent_customers"]
    ].copy()
    pair_features = (["customer_id", "article_id"], clw_pairs_df)

    return cust_last_week_cand, cust_last_week_pair_cand, features, pair_features


def create_popular_article_cand(
    transactions_df, customers_df, num_weeks, num_articles=12
):
    last_week_number = transactions_df["week_number"].max()

    article_purchases_df = (
        transactions_df.query(f"week_number >= {last_week_number - num_weeks + 1}")
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

    recent_popular_articles = article_purchases_df.index[:num_articles]
    popular_articles_df = cudf.DataFrame(
        {"article_id": cudf.Series(recent_popular_articles), "join_col": 1}
    )

    popular_articles_cand = cudf.DataFrame(
        {"customer_id": customers_df["customer_id"], "join_col": 1}
    )
    popular_articles_cand = popular_articles_cand.merge(
        popular_articles_df, on="join_col"
    )
    del popular_articles_cand["join_col"]

    features = (["article_id"], article_purchases_df)

    return popular_articles_cand, features


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
