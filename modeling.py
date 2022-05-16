import copy
import pickle as pkl
from datetime import datetime
import cudf
from cuml import ForestInference
import pandas as pd
import numpy as np
import lightgbm
from lightgbm import LGBMRanker
from sklearn.metrics import roc_auc_score


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
    X = X.to_pandas().astype("float32")
    y = y.to_pandas().astype("float32")

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
    X = X.to_pandas().astype("float32")

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


def full_cv_run(t, c, a, cand_features_func, score_func, **kwargs):
    # load training data
    train_eval_dfs = prepare_train_eval_modeling_dfs(
        t, c, a, cand_features_func, **kwargs
    )
    (
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
    ) = train_eval_dfs

    model = LGBMRanker(**(kwargs["lgbm_params"]), seed=42)

    eval_set = [(train_X, train_y), (eval_X, eval_y)]
    eval_group = [train_group_lengths, eval_group_lengths]
    eval_names = ["train", "validation"]

    le_callback = lightgbm.log_evaluation(kwargs["log_evaluation"])
    es_callback = lightgbm.early_stopping(kwargs["early_stopping"])

    model.fit(
        train_X,
        train_y,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_group=eval_group,
        eval_metric="MAP",
        eval_at=kwargs["eval_at"],
        callbacks=[le_callback, es_callback],
        group=train_group_lengths,
    )

    # get feature importance before save/reloading model
    feature_importance_dict = dict(
        zip(list(eval_X.columns), list(model.feature_importances_))
    )
    feature_importance_series = cudf.Series(feature_importance_dict).sort_values()

    # save/reload model
    model_path = f"model_{kwargs['label_week']}"
    model.booster_.save_model(model_path)
    model = ForestInference.load(model_path, model_type="lightgbm", output_class=False)

    # train predictions and scores
    train_pred = model.predict(train_X)
    print("Train AUC {:.4f}".format(roc_auc_score(train_y, train_pred)))
    print("Train score: ", score_func(train_ids_df, train_pred, train_truth_df))

    del train_X, train_y, train_group_lengths, train_truth_df, train_pred

    # evaluation predictions and scores
    eval_pred = pred_in_batches(model, eval_X)

    print("Eval AUC {:.4f}".format(roc_auc_score(eval_y, eval_pred)))
    eval_score = score_func(eval_ids_df, eval_pred, eval_truth_df)
    print("Eval score:", eval_score)

    # print feature importances
    print("\n")
    print(feature_importance_series)

    return eval_score


def run_all_cvs(
    t, c, a, cand_features_func, score_func, cv_weeks=[102, 103, 104], **params
):
    cv_scores = []
    total_duration = datetime.now() - datetime.now()

    for cv_week in cv_weeks:
        starting_time = datetime.now()

        cv_params = copy.deepcopy(params)
        cv_params.update({"label_week": cv_week})
        cv_score = full_cv_run(t, c, a, cand_features_func, score_func, **cv_params)

        cv_scores.append(cv_score)
        duration = datetime.now() - starting_time
        total_duration += duration
        print(f"Finished cv of week {cv_week} in {duration}. Score: {cv_score}\n")

    average_scores = round(np.mean(cv_scores), 5)
    print(
        f"Finished all {len(cv_weeks)} cvs in {total_duration}. "
        f"Average cv score: {average_scores}"
    )

    return average_scores


def full_sub_train_run(t, c, a, cand_features_func, score_func, **kwargs):
    (
        train_ids_df,
        train_X,
        train_group_lengths,
        train_y,
        train_truth_df,
    ) = prepare_concat_train_modeling_dfs(t, c, a, cand_features_func, **kwargs)

    model = LGBMRanker(**(kwargs["lgbm_params"]), seed=42)

    eval_set = [(train_X, train_y)]
    eval_group = [train_group_lengths]
    eval_names = ["train"]

    le_callback = lightgbm.log_evaluation(kwargs["log_evaluation"])

    model.fit(
        train_X,
        train_y,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_group=eval_group,
        eval_metric="MAP",
        eval_at=kwargs["eval_at"],
        callbacks=[le_callback],
        group=train_group_lengths,
    )

    # save/reload model
    model_path = f"model_{kwargs['label_week']}"
    model.booster_.save_model(model_path)
    model = ForestInference.load(model_path, model_type="lightgbm", output_class=False)

    # train predictions and scores
    train_pred = model.predict(train_X)
    print("Train AUC {:.4f}".format(roc_auc_score(train_y, train_pred)))
    print("Train score: ", score_func(train_ids_df, train_pred, train_truth_df))

    del train_X, train_y, train_group_lengths, train_truth_df, train_pred


def full_sub_predict_run(t, c, a, cand_features_func, **kwargs):
    customer_batches = []

    num_customers = len(c)
    third_point = num_customers // 3
    two_thirds_point = third_point * 2

    customer_batches.append(c[:third_point]["customer_id"].to_pandas().to_list())
    customer_batches.append(
        c[third_point:two_thirds_point]["customer_id"].to_pandas().to_list()
    )
    customer_batches.append(c[two_thirds_point:]["customer_id"].to_pandas().to_list())

    batch_preds = []
    for idx, customer_batch in enumerate(customer_batches):
        print(
            f"generating candidates/features for batch #{idx+1} of {len(customer_batches)}"
        )
        sub_ids_df, sub_X = prepare_prediction_dfs(
            t, c, a, cand_features_func, customer_batch=customer_batch, **kwargs
        )

        print(
            f"candidate/features shape of batch: ({sub_X.shape[0]:,}, {sub_X.shape[1]})",
        )

        prediction_models = kwargs.get("prediction_models")
        model_nums = len(prediction_models)

        first_model_path = prediction_models[0]
        first_model = ForestInference.load(
            first_model_path, model_type="lightgbm", output_class=False
        )

        print(f"predicting with '{first_model_path}'")
        sub_pred = pred_in_batches(first_model, sub_X) / model_nums
        del first_model

        for model_path in prediction_models[1:]:
            model = ForestInference.load(
                model_path, model_type="lightgbm", output_class=False
            )
            print(f"predicting with '{model_path}'")
            sub_pred2 = pred_in_batches(model, sub_X)
            del model
            sub_pred += sub_pred2 / model_nums

        batch_preds.append(create_predictions(sub_ids_df, sub_pred))

        del sub_ids_df, sub_X, sub_pred

    predictions = cudf.concat(batch_preds)

    return predictions


def categorical_cols_to_float(df):
    """
    to support using cuML's FIL
    returns the idxs so they can be passed to the model
    """
    col_names = list(df.columns)
    categorical_col_names = list(df.select_dtypes(include=["category"]).columns)

    categorical_col_idxs = []
    for col_name in categorical_col_names:
        categorical_col_idxs.append(col_names.index(col_name))
        df[col_name] = df[col_name].cat.codes.astype("float32")

    return categorical_col_idxs
