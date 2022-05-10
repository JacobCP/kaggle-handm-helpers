import pandas as pd
import cudf


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


def create_pairs(transactions_df, week_number, pairs_per_item, verbose=True):
    # get the dfs
    working_t_df = transactions_df[["customer_id", "article_id", "week_number"]].copy()

    # we'll look for pairs in last week only
    working_t_df = working_t_df.query(f"week_number <= {week_number}").copy()
    pairs_t_df = working_t_df.query(f"week_number == {week_number}").copy()
    pairs_t_df.columns = ["customer_id", "pair_article_id", "week_number"]

    # drop week number, and drop duplicates
    del working_t_df["week_number"], pairs_t_df["week_number"]
    working_t_df = working_t_df.drop_duplicates()
    pairs_t_df = pairs_t_df.drop_duplicates()

    # setup the loop
    unique_articles = working_t_df["article_id"].unique()
    batch_size = 5000
    batch_pairs_dfs = []

    # start the loop
    for i in range(0, len(unique_articles), batch_size):
        if verbose:
            print(f"processing article #{i:,} to #{i+batch_size:,}")

        # take batch of articles/transactions
        batch_articles = unique_articles[i : i + batch_size]
        batch_t = working_t_df[working_t_df["article_id"].isin(batch_articles)]

        # get total # of customers who bought each article (for later statistics)
        all_cust_counts = batch_t.groupby("article_id")["customer_id"].nunique()
        all_cust_counts = all_cust_counts.reset_index()
        all_cust_counts.columns = ["article_id", "all_customer_counts"]
        all_cust_counts["all_customer_counts"] -= 1  # not him himself

        # get all pairs for those articles (other articles those customers bought)
        batch_pairs_df = batch_t.merge(pairs_t_df, on="customer_id")

        # delete same-article pairs
        same_article_row_idxs = batch_pairs_df.query(
            "article_id==pair_article_id"
        ).index
        batch_pairs_df = batch_pairs_df.drop(same_article_row_idxs)

        # delete single customer articles
        c1s = (
            batch_pairs_df.groupby("article_id")[["customer_id"]]
            .nunique()
            .query("customer_id==1")
            .index
        )
        single_customer_row_idxs = batch_pairs_df[
            batch_pairs_df["article_id"].isin(c1s)
        ].index
        batch_pairs_df = batch_pairs_df.drop(single_customer_row_idxs)

        # get sorted counts of article-pair occurences
        batch_pairs_df = batch_pairs_df.groupby(["article_id", "pair_article_id"])[
            ["customer_id"]
        ].count()
        batch_pairs_df.columns = ["customer_count"]
        batch_pairs_df = batch_pairs_df.reset_index()
        batch_pairs_df = batch_pairs_df.sort_values(
            ["article_id", "customer_count"], ascending=False
        )

        # get top x pairs for each article
        batch_pairs_df = cudf_groupby_head(batch_pairs_df, "article_id", pairs_per_item)

        # calculate percentage statistic
        batch_pairs_df = batch_pairs_df.merge(all_cust_counts, on="article_id")
        batch_pairs_df["percent_customers"] = (
            batch_pairs_df["customer_count"] / batch_pairs_df["all_customer_counts"]
        )
        del batch_pairs_df["all_customer_counts"]

        batch_pairs_dfs.append(batch_pairs_df)

    all_article_pairs_df = cudf.concat(batch_pairs_dfs)

    return all_article_pairs_df
