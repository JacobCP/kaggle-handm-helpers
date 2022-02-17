import math
from datetime import timedelta

import pandas as pd


def day_week_numbers(dates: pd.Series):
    """
    assign week numbers to dates, such that:
    - week numbers represent consecutive actual weeks on the calendar
    - the latest date in the dates provided is the last day of the latest week number

    args:
    dates: pd.Series of date strings of pd.datetime objects

    returns:
    day_weeks: pd.Series of week numbers each day in original pd.Series is in

    """
    all_dates = pd.to_datetime(dates)
    numbered_days = all_dates - all_dates.min() + timedelta(1)
    numbered_days = numbered_days.apply(lambda x: x.days)
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = (numbered_days / 7).apply(math.ceil)

    return day_weeks
