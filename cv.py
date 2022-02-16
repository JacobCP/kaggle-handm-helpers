from datetime import datetime, timedelta
import pandas as pd
from typing import Union


def ground_truth(transactions_df):
    customer_trans = transactions_df[["customer_id", "article_id"]]
    customer_trans = customer_trans.groupby("customer_id")[["article_id"]].agg(set)
    customer_trans.columns = ["prediction"]
    customer_trans["prediction"] = customer_trans["prediction"].apply(list)
    customer_trans["prediction"] = customer_trans["prediction"].apply(
        lambda x: " ".join(x)
    )
    gt = customer_trans.reset_index()

    return gt


def train_validation_split(
    transactions_df: pd.DataFrame, validation_date: Union[str, pd.Timestamp]
):
    """
    split transaction_df into train/validation
    validation_df will be week starting on Wednesday on or before date provided

    (because test week starts on a Wednesday)
    """

    # get the wednesday before for start_date
    validation_date = pd.to_datetime(validation_date)
    weekday = datetime.weekday(validation_date)
    weekday += 7  # so we don't have negatives to deal with
    validation_start_date = validation_date - timedelta((weekday - 2) % 7)

    # end date is 6 days later
    validation_end_date = validation_start_date + timedelta(6)

    # convert back to yyyy-mm-dd formatted strings
    validation_start_date = validation_start_date.strftime("%Y-%m-%d")
    validation_end_date = validation_end_date.strftime("%Y-%m-%d")

    # use dates to filter/create train/validation dfs
    train_df = transactions_df.query(f"t_dat < '{validation_start_date}'").copy()
    validation_df = transactions_df.query(
        f"t_dat >= '{validation_start_date}' & t_dat <= '{validation_end_date}'"
    ).copy()

    print(
        f"Validation week is Wednesday, {validation_start_date} through Tuesday, {validation_end_date}"
    )

    return train_df, validation_df


def comp_average_precision(
    data_true: pd.Series,
    data_predicted: pd.Series,
) -> float:
    """
    :param data_true: items that were actually purchased by user
    :param data_predicted: items we recommended to user
    """

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
