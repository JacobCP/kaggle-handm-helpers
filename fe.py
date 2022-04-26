import math
import pickle as pkl
from datetime import timedelta

import numpy as np
import pandas as pd
import cudf


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


def day_numbers(dates: cudf.Series):
    """
    assign consecutive number to dates, such that earliest date is 0

    args:
    dates: pd.Series of date strings

    returns:
    all_day_numbers: pd.Series of day numbers each day in original pd.Series corresponds to

    """

    unique_dates = dates.unique()
    unique_dates = unique_dates.to_pandas()
    unique_dates = np.sort(unique_dates)
    number_range = np.arange(len(unique_dates))
    date_number_dict = dict(zip(unique_dates, number_range))

    all_day_numbers = dates.map(date_number_dict)
    all_day_numbers = all_day_numbers.astype("int16")

    return all_day_numbers


def day_week_numbers(dates: cudf.Series):
    """
    assign week numbers to dates, such that:
    - week numbers represent consecutive actual weeks on the calendar
    - the latest date in the dates provided is the last day of the latest week number

    args:
    dates: pd.Series of date strings

    returns:
    day_weeks: pd.Series of week numbers each day in original pd.Series is in

    """
    pd_dates = cudf.to_datetime(dates)

    unique_dates = cudf.Series(pd_dates.unique())
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = (numbered_days / 7).applymap(lambda x: math.ceil(x))
    day_weeks_map = cudf.DataFrame(
        {"day_weeks": day_weeks, "unique_dates": unique_dates}
    ).set_index("unique_dates")["day_weeks"]
    all_day_weeks = pd_dates.map(day_weeks_map)
    all_day_weeks = all_day_weeks.astype("int8")

    return all_day_weeks


def year_week_numbers(weeks: cudf.Series):
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


def how_many_ago(sequential_numbers: cudf.Series):
    """
    given a pd_or_cudf.series of numbers between 0 and n,
    returns new series with n subtracted from each element

    (so that if n reflects last day or week number, output will reflect how many days or weeks
    earlier each element was from last day or week, with -1 meaning 1 day or week earlier etc.)
    """

    return sequential_numbers - sequential_numbers.max()


def create_cust_hier_features(transactions_df, articles_df, hier_cols, features_db):
    sample_col = "t_dat"

    # create hiers
    for hier_col in hier_cols:
        # total customer counts
        total_cust_counts = transactions_df.groupby("customer_id")[sample_col].count()

        # add hierarchy column to transactions
        article_hier_lookup = articles_df.set_index("article_id")[hier_col]
        transactions_df[hier_col] = transactions_df["article_id"].map(
            article_hier_lookup
        )

        # get customer/hierarchy statistics
        cust_hier = (
            transactions_df.groupby(["customer_id", hier_col])[sample_col]
            .count()
            .reset_index()
        )
        cust_hier.columns = list(cust_hier.columns)[:-1] + ["cust_hier_counts"]
        cust_hier = cust_hier.sort_values(
            ["customer_id", "cust_hier_counts"], ascending=False
        )

        cust_hier["total_counts"] = cust_hier["customer_id"].map(total_cust_counts)

        hier_portion_column = f"cust_{hier_col}_portion"
        cust_hier[hier_portion_column] = (
            cust_hier["cust_hier_counts"] / cust_hier["total_counts"]
        )
        cust_hier = cust_hier[["customer_id", hier_col, hier_portion_column]]
        cust_hier = cust_hier.set_index(["customer_id", hier_col])
        cust_hier[hier_portion_column] = cust_hier[hier_portion_column].astype(
            "float32"
        )
        features_db[hier_portion_column] = (["customer_id", hier_col], cust_hier)
