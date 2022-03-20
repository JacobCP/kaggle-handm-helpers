def create_pairs(
    df,
    last_week_number,
):
    # get only the columns we need
    df = df[["customer_id", "article_id", "week_number"]].copy()

    # drop everything after
    df = df.query(f"week_number <= {last_week_number}").copy()

    # we'll look for pairs in last week
    last_week_df = df.query(f"week_number=={last_week_number}").copy()

    # no longer need week number
    del df["week_number"], last_week_df["week_number"]

    # drop articles with only one customer
    article_customer_counts = df.groupby("article_id")["customer_id"].nunique()
    single_customer_articles = article_customer_counts[
        article_customer_counts == 1
    ].index
    df = df[~(df["article_id"].isin(single_customer_articles))].copy()

    # drop duplicates
    df, last_week_df = df.drop_duplicates(), last_week_df.drop_duplicates()

    # FIND ITEMS PURCHASED TOGETHER
    vc = df.article_id.value_counts()
    pairs = {}
    for j, i in enumerate(vc.index.values):
        if j % 50 == 0:
            print(j, ", ", end="")
        USERS = df.loc[df.article_id == i.item(), "customer_id"].unique()
        vc2 = last_week_df.loc[
            (last_week_df.customer_id.isin(USERS))
            & (last_week_df.article_id != i.item()),
            "article_id",
        ]
        if len(vc2) == 0:
            continue
        try:
            vc2 = vc2.value_counts()
            pairs[i.item()] = vc2.index[0]
        except Exception as e:
            print(type(e), ":", e)

    return pairs
