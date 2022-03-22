import pandas as pd
import torch

if torch.cuda.is_available():
    import cudf  # type: ignore

    gpu_available = True
    pd_or_cudf = cudf
else:
    gpu_available = False
    pd_or_cudf = pd


def create_pairs(
    df,
    last_week_number,
    pairs_per_item,
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
    df, lw_df = df.drop_duplicates(), last_week_df.drop_duplicates()

    # FIND ITEMS PURCHASED TOGETHER
    vc = df.article_id.value_counts()
    pairs_dfs = []
    for j, i in enumerate(vc.index.values):
        if j % 50 == 0:
            print(j, ", ", end="")

        USERS = df.loc[df.article_id == i.item(), "customer_id"].unique()
        row_filter = (lw_df.customer_id.isin(USERS)) & (lw_df.article_id != i.item())
        pairs_df = lw_df.loc[row_filter, "article_id"]

        if len(pairs_df) == 0:
            continue

        try:
            pairs_df = pairs_df.value_counts().reset_index()[:pairs_per_item]
            pairs_df.columns = ["pair_article_id", "customer_count"]
            pairs_df["percent_customers"] = pairs_df["customer_count"] / len(USERS)
            pairs_df["article_id"] = i.item()

            pairs_dfs.append(pairs_df)
        except Exception as e:
            print(type(e), ":", e)

    # concatenate them all together
    all_pairs_df = pd_or_cudf.concat(pairs_dfs).reset_index(drop=True)

    return all_pairs_df
