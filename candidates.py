def create_candidates(transactions_df, customers_df, articles_df):
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
    recent_popular_items = article_purchases_df.index[:12]

    popular_items_cand = list(
        [
            customer_id + "_" + article_id
            for customer_id in customers_df["customer_id"]
            for article_id in recent_popular_items
        ]
    )
    features["article_id"] = article_purchases_df
    candidates = past_purchases_cand + popular_items_cand

    return candidates, features


def filter_candidates(candidates):
    """
    manual filtering logic of candidates, to keep it from getting out of hand
    """
    return candidates
