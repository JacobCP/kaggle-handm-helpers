import cudf
import pandas as pd


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


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
    transactions_df,
    customers_df,
    articles_df,
    num_weeks,
    hier_col,
    num_candidates,
    num_articles=12,
):
    ###########################################
    # first get general popular candidates
    ###########################################
    last_week_number = transactions_df["week_number"].max()

    # baseline
    article_purchases_df = (
        transactions_df.query(f"week_number >= {last_week_number - num_weeks + 1}")
        .groupby("article_id")["customer_id"]
        .count()
        .sort_values(ascending=False)
    )
    article_purchases_df = article_purchases_df.reset_index()
    article_purchases_df.columns = ["article_id", "counts"]
    popular_articles_df = article_purchases_df[:num_candidates].copy()
    popular_articles_df["join_col"] = 1

    popular_articles_cand = cudf.DataFrame(
        {"customer_id": customers_df["customer_id"], "join_col": 1}
    )
    popular_articles_cand = popular_articles_cand.merge(
        popular_articles_df, on="join_col"
    )
    del popular_articles_cand["join_col"]

    ###################################################
    # now let's limit it by cust/hierarchy information
    ###################################################
    sample_col = "t_dat"

    # add hierarchy column to transactions
    transactions_df[hier_col] = transactions_df["article_id"].map(
        articles_df.set_index("article_id")[hier_col]
    )
    # get customer/hierarchy statistics
    cust_hier = (
        transactions_df.groupby(["customer_id", hier_col])[sample_col]
        .count()
        .reset_index()
    )
    cust_hier.columns = list(cust_hier.columns)[:-1] + ["cust_hier_counts"]
    cust_hier = cust_hier.sort_values(
        ["customer_id", "cust_hier_counts"], ascending=False
    )
    cust_hier["total_counts"] = cust_hier["customer_id"].map(
        transactions_df.groupby("customer_id")[sample_col].count()
    )
    cust_hier["cust_hier_portion"] = (
        cust_hier["cust_hier_counts"] / cust_hier["total_counts"]
    )
    cust_hier = cust_hier[["customer_id", hier_col, "cust_hier_portion"]].copy()

    # add customer/hierarchy statistics to candidates
    popular_articles_cand[hier_col] = popular_articles_cand["article_id"].map(
        articles_df.set_index("article_id")[hier_col]
    )
    popular_articles_cand = popular_articles_cand.merge(
        cust_hier, on=["customer_id", hier_col], how="left"
    )
    popular_articles_cand["cust_hier_portion"] = popular_articles_cand[
        "cust_hier_portion"
    ].fillna(-1)

    del popular_articles_cand[hier_col]

    # take top based on customer/hierarchy statistics
    popular_articles_cand = popular_articles_cand.sort_values(
        ["customer_id", "cust_hier_portion", "counts"], ascending=False
    )
    popular_articles_cand = popular_articles_cand[["customer_id", "article_id"]].copy()
    popular_articles_cand = cudf_groupby_head(popular_articles_cand, "customer_id", 12)
    popular_articles_cand = popular_articles_cand.sort_values(
        ["customer_id", "article_id"]
    )
    popular_articles_cand = popular_articles_cand.reset_index(drop=True)

    # save the customer/hierarchy statistics in the features
    cust_hier_features = cust_hier.set_index(["customer_id", hier_col])
    cust_hier_features = cust_hier_features.reset_index().set_index(
        ["customer_id", hier_col]
    )
    cust_hier_features = (["customer_id", hier_col], cust_hier_features)

    # and save the article purchase statistics
    article_purchases_df = article_purchases_df[["article_id", "counts"]]
    article_purchases_df.columns = ["article_id", "recent_popularity_counts"]
    article_purchase_features = (
        ["article_id"],
        article_purchases_df.set_index("article_id"),
    )

    return popular_articles_cand, cust_hier_features, article_purchase_features


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
