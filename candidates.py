import pandas as pd


def create_candidates(transactions_df, customers_df, articles_df, **kwargs):
    """
    will return candidates in form customer_id_article_id
    will also return 'features' dict, where:
        - format is {<column_name>:<features_df},
        - column_name is a column in articles, customers or transactions
        - features_df.index values correspond to values of column_name column
        - columns of features_df are engineered features unique to the df.index values
    """
    features = {}

    # past customer/article purchases
    past_purchases_df = (
        transactions_df.groupby("customer_article")
        .agg(
            last_cust_art_purchase_week=("week_number", "max"),
            cust_art_purchase_count=("article_id", "count"),
        )
        .sort_values("cust_art_purchase_count", ascending=False)
    )

    features["customer_article"] = past_purchases_df
    past_purchases_cand = list(
        past_purchases_df.reset_index()["customer_article"].drop_duplicates()
    )

    # recently popular items
    article_purchases_df = (
        transactions_df.groupby("article_id")
        .agg(
            last_art_purchase_week=("week_number", "max"),
            art_purchase_count=("article_id", "count"),
        )
        .sort_values("art_purchase_count", ascending=False)
    )
    num_recent_items = kwargs.get("num_recent_items", 12)
    recent_popular_items = article_purchases_df.index[:num_recent_items]

    popular_items_cand = list(
        [
            (customer_id, article_id)
            for customer_id in customers_df["customer_id"]
            for article_id in recent_popular_items
        ]
    )
    features["article_id"] = article_purchases_df
    candidates = list(set(past_purchases_cand + popular_items_cand))

    return candidates, features


def filter_candidates(candidates):
    """
    manual filtering logic of candidates, to keep it from getting out of hand
    """
    return candidates


def add_features_to_candidates(candidates, features, customers_df, articles_df):
    """
    expands candidates to have customer_id and article_id fields,
    adds fields needed to merge in features,
    and merges features in
    """
    candidates_df = pd.DataFrame({"customer_article": candidates})
    candidates_df["customer_id"] = candidates_df["customer_article"].apply(
        lambda x: x[0]
    )
    candidates_df["article_id"] = candidates_df["customer_article"].apply(
        lambda x: x[1]
    )

    for col_name in features:
        feature_df = features[col_name]

        # add the key to our df so we can merge the features in
        if col_name not in candidates_df:
            if col_name in customers_df:
                col_name_dict = customers_df.set_index("customer_id")[col_name]
                candidates_df[col_name] = candidates_df["customer_id"].map(
                    col_name_dict
                )
            elif col_name in articles_df:
                col_name_dict = articles_df.set_index("article_id")[col_name]
                candidates_df[col_name] = candidates_df["article_id"].map(col_name_dict)

        # now we can add the features
        candidates_df = candidates_df.merge(feature_df, how="left", on=col_name)

    return candidates_df
