import cudf
import pandas as pd


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


def create_recent_customer_candidates(
    transactions_df, recent_customer_weeks, customers=None
):
    if customers is not None:
        transactions_df = transactions_df[
            transactions_df["customer_id"].isin(customers)
        ]

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


def create_last_customer_weeks_and_pairs(
    transactions_df, article_pairs_df, num_weeks, num_pair_weeks, customers
):
    clw_df = transactions_df[["customer_id", "article_id", "t_dat"]].copy()
    if customers is not None:
        clw_df = clw_df[clw_df["customer_id"].isin(customers)]

    # only transactions in "x" weeks before last customer purchase
    last_customer_purchase_dat = clw_df.groupby("customer_id")["t_dat"].max()
    clw_df["max_cust_dat"] = clw_df["customer_id"].map(last_customer_purchase_dat)
    clw_df["sample"] = 1

    clw_df = (
        clw_df.groupby(["customer_id", "article_id"])
        .agg(
            {
                "max_cust_dat": "max",
                "sample": "count",
                "t_dat": "max",
            }
        )
        .rename(
            columns={
                "max_cust_dat": "last_c_purchase_date",
                "sample": "ca_count",
                "t_dat": "last_ca_purchase_date",
            }
        )
        .reset_index()
    )
    clw_df["last_ca_purchase_diff"] = (
        clw_df["last_c_purchase_date"] - clw_df["last_ca_purchase_date"]
    )

    clw_pairs_df = clw_df.query(
        f"last_ca_purchase_diff <= {num_pair_weeks * 7 - 1}"
    ).copy()
    clw_df = clw_df.query(f"last_ca_purchase_diff <= {num_weeks * 7 - 1}").copy()

    del last_customer_purchase_dat

    # merge with pairs, and get max of:
    #  - sources' last week(s) purchase count
    #  - count and percent of customer pairs (see generating code for details)
    clw_pairs_df = clw_pairs_df.merge(article_pairs_df, on="article_id")

    clw_pairs_df = (
        clw_pairs_df.groupby(["customer_id", "pair_article_id"])[
            [
                "ca_count",
                "last_ca_purchase_date",
                "last_ca_purchase_diff",
                "customer_count",
                "percent_customers",
            ]
        ]
        .max()
        .reset_index()
    )
    clw_pairs_df.columns = [
        "customer_id",
        "article_id",
        "pair_ca_count",
        "pair_last_ca_purchase_date",
        "pair_last_ca_purchase_diff",
        "pair_customer_count",
        "pair_percent_customers",
    ]
    clw_pairs_df = clw_pairs_df.query("pair_customer_count > 2").copy()

    cust_last_week_cand = clw_df[["customer_id", "article_id"]].drop_duplicates()
    cust_last_week_pair_cand = clw_pairs_df[
        ["customer_id", "article_id"]
    ].drop_duplicates()

    clw_df = clw_df.set_index(["customer_id", "article_id"])[
        ["ca_count", "last_ca_purchase_date", "last_ca_purchase_diff"]
    ].copy()
    features = (["customer_id", "article_id"], clw_df)

    clw_pairs_df = clw_pairs_df.set_index(["customer_id", "article_id"])[
        [
            "pair_ca_count",
            "pair_last_ca_purchase_date",
            "pair_last_ca_purchase_diff",
            "pair_customer_count",
            "pair_percent_customers",
        ]
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
    customers=None,
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

    # from here on, only care about relevant customers
    if customers is not None:
        transactions_df = transactions_df[
            transactions_df["customer_id"].isin(customers)
        ]
        customers_df = customers_df[customers_df["customer_id"].isin(customers)]

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
    popular_articles_cand = cudf_groupby_head(
        popular_articles_cand, "customer_id", num_articles
    )
    popular_articles_cand = popular_articles_cand.sort_values(
        ["customer_id", "article_id"]
    )
    popular_articles_cand = popular_articles_cand.reset_index(drop=True)

    # and save the article purchase statistics
    article_purchases_df = article_purchases_df[["article_id", "counts"]]
    article_purchases_df.columns = ["article_id", "recent_popularity_counts"]
    article_purchase_features = (
        ["article_id"],
        article_purchases_df.set_index("article_id"),
    )

    return popular_articles_cand, article_purchase_features


def create_age_bucket_candidates(
    transactions_df, customers_df, age_buckets, customers=None, articles=12
):
    # get transactions we're working with
    working_t_df = transactions_df.copy()
    working_t_df = working_t_df.drop_duplicates(
        ["customer_id", "article_id", "week_number"]
    )

    # create the buckets
    buckets_df = (
        customers_df[["customer_id"]].drop_duplicates().set_index("customer_id")
    )
    buckets_df["age"] = customers_df.set_index("customer_id").age
    buckets_df["age_bucket"] = pd.qcut(
        buckets_df["age"].to_pandas(), age_buckets
    ).cat.codes

    # choose bucket
    selected_buckets = ["age_bucket"]

    # add the buckets to the transactions
    working_t_df = working_t_df.merge(
        buckets_df[selected_buckets].reset_index(), on="customer_id"
    )

    # get the popularity
    last_week = working_t_df["week_number"].max()
    pi_df = (
        working_t_df.query(f"week_number=={last_week}")
        .groupby(selected_buckets + ["article_id"])["t_dat"]
        .count()
        .reset_index()
        .sort_values(selected_buckets + ["t_dat"], ascending=False)
    )
    pi_df = cudf_groupby_head(pi_df, selected_buckets, articles)

    # candidates - merge customer with their bucket
    can_df = buckets_df.reset_index()[["customer_id"] + selected_buckets].merge(
        pi_df, on=selected_buckets
    )
    can_df.columns = ["customer_id", "age_bucket", "article_id", "article_bucket_count"]

    # features dfs
    buckets_df = buckets_df[["age_bucket"]].copy()
    bucket_counts_df = can_df[
        ["customer_id", "article_id", "article_bucket_count"]
    ].copy()
    bucket_counts_df = bucket_counts_df.set_index(["customer_id", "article_id"])

    # candidates_df
    can_df = can_df[["customer_id", "article_id"]]
    if customers is not None:
        can_df = can_df[can_df["customer_id"].isin(customers)]

    return (
        can_df,
        (["customer_id"], buckets_df),
        (["customer_id", "article_id"], bucket_counts_df),
    )


def add_features_to_candidates(candidates_df, features, customers_df, articles_df):
    """
    adds fields needed to merge in features
    and merges features in
    """
    for features_key in features:
        col_names, feature_df = features[features_key]

        # add the key to our df so we can merge the features in
        to_delete = []
        for col_name in col_names:
            if col_name not in candidates_df:
                if col_name in customers_df:
                    col_name_dict = customers_df.set_index("customer_id")[col_name]
                    candidates_df[col_name] = candidates_df["customer_id"].map(
                        col_name_dict
                    )
                    to_delete.append(col_name)
                elif col_name in articles_df:
                    col_name_dict = articles_df.set_index("article_id")[col_name]
                    candidates_df[col_name] = candidates_df["article_id"].map(
                        col_name_dict
                    )
                    to_delete.append(col_name)

        # now we can add the features
        candidates_df = candidates_df.merge(feature_df, how="left", on=col_names)

        for col_name in to_delete:
            del candidates_df[col_name]

    return candidates_df


def filter_candidates(candidates, transactions_df, **kwargs):
    recent_art_weeks = kwargs["filter_recent_art_weeks"]
    recent_articles = transactions_df.query(
        f"week_number >= {kwargs['label_week'] - recent_art_weeks}"
    )["article_id"]

    num_articles = kwargs.get("filter_num_articles", None)
    if num_articles is None:
        recent_articles = recent_articles.drop_duplicates()
    else:
        recent_item_counts = recent_articles.value_counts()
        most_popular_items = recent_item_counts[:num_articles].index
        most_popular_items = most_popular_items.to_pandas().to_list()
        recent_articles = most_popular_items

    candidates = candidates[candidates["article_id"].isin(recent_articles)].copy()

    return candidates
