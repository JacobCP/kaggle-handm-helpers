import copy
import cudf
import pandas as pd
import numpy as np


def get_group_lengths(df):
    """
    Get group_lengths to pass to LGBMRanker
    """
    df = df.to_pandas()

    return list(df.groupby("customer_id")["article_id"].count())


def cudf_groupby_head(df, groupby, head_count):
    df = df.to_pandas()

    head_df = df.groupby(groupby).head(head_count)

    head_df = cudf.DataFrame(head_df)

    return head_df


def create_predictions(ids_df, preds):
    ids_df["pred"] = preds
    ids_df = ids_df.sort_values(["customer_id", "pred"], ascending=False)
    ids_df = cudf_groupby_head(ids_df, "customer_id", 12)
    predictions = ids_df.groupby("customer_id")["article_id"].agg(list)

    return predictions


def pred_in_batches(model, features_df, batch_size=1000000):
    preds = []
    num_batches = int(len(features_df) / 1000000) + 1
    for batch in range(num_batches):
        batch_df = features_df.iloc[batch * batch_size : (batch + 1) * batch_size, :]
        preds.append(model.predict(batch_df))

    return np.concatenate(preds)


def prep_cudf_to_pandas(cudf_df, inplace=False):
    """
    prepared a cudf df for efficient conversion to pandas
    by converting all int columns with null values to float32
    (otherwise they'd become float 64 in pandas)

    if inplace=True, will convert inplace and return None
    if inplace=False, will create copy, convert and return

    (meant for use on cudf df, but won't fail on pandas df)
    """
    if not inplace:
        working_df = cudf_df.copy()
    else:
        working_df = cudf_df

    for col_name in working_df:
        is_int_type = str(working_df[col_name].dtype)[:3] == "int"
        has_nulls = working_df[col_name].isna().mean() > 0
        if is_int_type and has_nulls:
            working_df[col_name] = working_df[col_name].astype("float32")

    if not inplace:
        return working_df
    else:
        return None


def prepare_modeling_dfs(t, c, a, cand_features_func, **params):
    # "cf" means -> candidates_with_features
    cf_df, label_df = cand_features_func(t, c, a, **params)

    # adding y column
    label_df = label_df[["customer_id", "article_id"]].drop_duplicates()
    label_df["match"] = 1
    cf_df = cf_df.merge(label_df, how="left", on=["customer_id", "article_id"])
    cf_df["match"] = cf_df["match"].fillna(0).astype("int8")
    cf_df = cf_df.sample(frac=1, random_state=42).reset_index(drop=True)
    del label_df["match"]

    # only customer with some positives ones
    customers_with_positives = cf_df.query("match==1")["customer_id"].unique()
    cf_df = cf_df[cf_df["customer_id"].isin(customers_with_positives)]
    cf_df = cf_df.sort_values("customer_id").reset_index(drop=True)

    # get group lengths
    group_lengths = get_group_lengths(cf_df)

    # split out ids
    ids_df = cf_df[["customer_id", "article_id"]].copy()
    del cf_df["customer_id"], cf_df["article_id"]

    # split out y
    y = cf_df[["match"]].copy()
    del cf_df["match"]

    # rename
    X = cf_df
    del cf_df

    ground_truth_df = label_df
    del label_df

    # prep for pandas
    prep_cudf_to_pandas(X, inplace=True)
    prep_cudf_to_pandas(y, inplace=True)
    X = X.to_pandas()
    y = y.to_pandas()

    y = y["match"]

    return ids_df, X, group_lengths, y, ground_truth_df


def prepare_prediction_dfs(t, c, a, cand_features_func, customer_batch=None, **params):
    # "cf" means -> candidates_with_features
    cf_df, _ = cand_features_func(t, c, a, customer_batch=customer_batch, **params)

    # split out ids
    ids_df = cf_df[["customer_id", "article_id"]].copy()
    del cf_df["customer_id"], cf_df["article_id"]

    # rename
    X = cf_df
    del cf_df

    # prep for pandas
    prep_cudf_to_pandas(X, inplace=True)
    X = X.to_pandas()

    return ids_df, X


def prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **params):
    working_params = copy.deepcopy(params)

    if params["num_concats"] > 1:
        empty_list = [None] * params["num_concats"]
        empty_lists = [empty_list.copy() for i in range(5)]

        (
            train_ids_df,
            train_X,
            train_group_lengths,
            train_y,
            train_truth_df,
        ) = empty_lists

        for i in range(params["num_concats"]):
            week_num = params["label_week"] - (i + 1)
            print(f"preparing training modeling dfs for {week_num}...")
            working_params["label_week"] = week_num
            (
                train_ids_df[i],
                train_X[i],
                train_group_lengths[i],
                train_y[i],
                train_truth_df[i],
            ) = prepare_modeling_dfs(t, c, a, cand_features_func, **working_params)

        print("concatenating all weeks together")
        train_ids_df = cudf.concat(train_ids_df)
        train_group_lengths = sum(train_group_lengths, [])
        train_truth_df = cudf.concat(train_truth_df)
        train_X = pd.concat(train_X)
        train_y = pd.concat(train_y)
    else:
        week_num = params["label_week"] - 1
        print(f"preparing training modeling dfs for {week_num}...")
        working_params["label_week"] = week_num
        (
            train_ids_df,
            train_X,
            train_group_lengths,
            train_y,
            train_truth_df,
        ) = prepare_modeling_dfs(t, c, a, cand_features_func, **working_params)

    return train_ids_df, train_X, train_group_lengths, train_y, train_truth_df


def prepare_train_eval_modeling_dfs(t, c, a, cand_features_func, **params):
    (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
    ) = prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **params)

    print("preparing evaluation modeling dfs...")
    (
        eval_ids_df,
        eval_X,
        eval_group_lengths,
        eval_y,
        eval_truth_df,
    ) = prepare_modeling_dfs(t, c, a, cand_features_func, **params)

    return (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
        eval_ids_df,
        eval_X,
        eval_group_lengths,
        eval_y,
        eval_truth_df,
    )
