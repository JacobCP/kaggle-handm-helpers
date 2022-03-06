import math
import pickle as pkl
from datetime import timedelta

import pandas as pd
import torch


if torch.cuda.is_available():
    import cudf  # type: ignore

    gpu_available = True
    apply_method_str = "applymap"
    pd_or_cudf = cudf
else:
    gpu_available = False
    apply_method_str = "apply"
    pd_or_cudf = pd


def my_apply(series, func_to_apply):
    apply_method = getattr(series, apply_method_str)

    return apply_method(func_to_apply)


def reduce_customer_id_memory(customers_df, other_dfs):
    """
    reduces memory in dfs in place

    returns path to pickled customer_id mapping, for submission use
    """
    ####
    # customer id compression

    # save the mapping we'll need to get back
    index_to_id_dict = customers_df["customer_id"]
    file_path = "id_to_index_dict.pkl"
    with open(file_path, "wb") as f:
        pkl.dump(index_to_id_dict, f)
    del index_to_id_dict

    # now compress
    id_to_index_dict = customers_df.reset_index().set_index("customer_id")["index"]
    customers_df["customer_id"] = (
        customers_df["customer_id"].map(id_to_index_dict).astype("int32")
    )

    for other_df in other_dfs:
        if "customer_id" in other_df:
            other_df["customer_id"] = (
                other_df["customer_id"].map(id_to_index_dict).astype("int32")
            )

    return file_path


def day_week_numbers(dates: pd_or_cudf.Series):
    """
    assign week numbers to dates, such that:
    - week numbers represent consecutive actual weeks on the calendar
    - the latest date in the dates provided is the last day of the latest week number

    args:
    dates: pd.Series of date strings of pd.datetime objects

    returns:
    day_weeks: pd.Series of week numbers each day in original pd.Series is in

    """
    pd_dates = pd_or_cudf.to_datetime(dates)

    unique_dates = pd_or_cudf.Series(pd_dates.unique())
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = my_apply(numbered_days / 7, lambda x: math.ceil(x))
    day_weeks_map = pd_or_cudf.DataFrame(
        {"day_weeks": day_weeks, "unique_dates": unique_dates}
    ).set_index("unique_dates")["day_weeks"]
    all_day_weeks = pd_dates.map(day_weeks_map)
    all_day_weeks = all_day_weeks.astype("int8")

    return all_day_weeks


def year_week_numbers(weeks: pd_or_cudf.Series):
    """
    convert consecutive week numbers into year and week features, such that:
    - each year has 52 weeks
    - the last week number is the series provided will be week 52 of the last year

    e.g. week number 106 will be week 2 of year 3

    args:
    weeks: pd.Series of integers, representing consecutive weeks

    returns:
    years: integers representing consecutive "years" of 52 weeks
    year_weeks: integers representing weeks within said "years"
    """

    years = (weeks / 52).apply(math.ceil)
    year_weeks = (weeks % 52).replace(0, 52)

    return years, year_weeks
