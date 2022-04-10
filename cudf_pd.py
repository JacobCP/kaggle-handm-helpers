import cudf


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df
