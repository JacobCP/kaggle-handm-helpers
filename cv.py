from typing import List

import pandas as pd
import cudf


def ground_truth(transactions_df):
    customer_trans = transactions_df[["customer_id", "article_id"]].drop_duplicates()
    customer_trans = customer_trans.groupby("customer_id")[["article_id"]].agg(list)
    customer_trans.columns = ["prediction"]

    gt = customer_trans.reset_index()

    return gt


def feature_label_split(
    transactions_df: cudf.DataFrame,
    label_week_number: int,
    feature_week_length=None,
):
    """
    split transaction_df into train/validation
    use week numbers created already using fe.day_week_numbers() function
    """

    # use week numbers to filter/create train/validation dfs
    label_df = transactions_df.query(f"week_number=={label_week_number}").copy()
    features_df = transactions_df.query(f"week_number < {label_week_number}").copy()
    if feature_week_length is not None:
        features_df = features_df.query(
            f"week_number >= {label_week_number - feature_week_length}"
        ).copy()

    return features_df, label_df


def report_candidates(candidates: List[str], ground_truth_candidates: List[str]):
    """
    Calculate recall and candidates factor (num(candidates) / num(ground_truth_candidates))
    """

    num_candidates = len(candidates)
    num_ground_truth = len(ground_truth_candidates)
    all_candidates = cudf.concat(
        [ground_truth_candidates, candidates]
    ).drop_duplicates()
    num_true_candidates = num_candidates + num_ground_truth - len(all_candidates)
    recall = num_true_candidates / num_ground_truth
    precision = num_true_candidates / num_candidates

    print(
        f"candidates recall: {recall:.2%} ({num_true_candidates:,}/{num_ground_truth:,})"
    )
    print(
        f"candidates precision: {precision:.2%} ({num_true_candidates:,}/{num_candidates:,})"
    )

    return recall, precision


def comp_average_precision(
    data_true: cudf.Series,
    data_predicted: cudf.Series,
) -> float:
    """
    :param data_true: items that were actually purchased by user
    :param data_predicted: items we recommended to user
    """
    data_true = data_true.to_pandas().apply(list)
    data_predicted = data_predicted.to_pandas().apply(list)

    # for this competition, we don't score ones without any purchases
    data_true = data_true[data_true.notna()]
    data_true = data_true[data_true.apply(len) > 0]

    if len(data_true) == 0:
        raise ValueError("data_true is empty")

    # convert to df so we can use `apply`
    eval_df = pd.DataFrame({"true": data_true})

    eval_df["predicted"] = data_predicted  # this way only get ones in data_true

    # replace na predictions with empty list
    eval_df["predicted"] = eval_df["predicted"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # getting the counts of true/predicted
    eval_df["n_items_true"] = eval_df["true"].apply(len)
    eval_df["n_items_predicted"] = eval_df["predicted"].apply(lambda x: min(len(x), 12))

    # ignore zero predicted (zero true is not even included...)
    non_zero_filter = eval_df["n_items_predicted"] > 0
    eval_df = eval_df[non_zero_filter].copy()

    def row_precision(items_true, items_predicted, n_items_predicted, n_items_true):
        n_correct_items = 0
        precision = 0.0

        for item_idx in range(n_items_predicted):
            if items_predicted[item_idx] in items_true:
                n_correct_items += 1
                precision += n_correct_items / (item_idx + 1)

        return precision / min(n_items_true, 12)

    eval_df["row_precision"] = eval_df.apply(
        lambda x: row_precision(
            x["true"], x["predicted"], x["n_items_predicted"], x["n_items_true"]
        ),
        axis=1,
    )

    return eval_df["row_precision"].sum() / len(data_true)
