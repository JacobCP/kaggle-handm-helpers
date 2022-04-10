from typing import List
import pandas as pd
import cudf


def sample_customers(
    customer_df: cudf.DataFrame,
    other_dfs: List[cudf.DataFrame],
    sample_size: int = None,
    sample_percent: float = 0.01,
    seed: int = 20,
):
    """
    sample a percentage or set number of customers
    and filter other data to those customer_ids

    args:
    customer_df: pd.DataFrame with customer info, and a customer_id column
    other_dfs: list of other pd.DataFrames with other info, with customer_id columns
    sample_size: int - number of customers to sample. Defaults to None, so that sample_percent is used
    sample_percent: float - ratio of customers to sample. defaults to 1 percent.
    seed: int - random state for reproducible sampling

    returns:
    sampled_customer_df - sample taken of customer_df arg
    sampled_other_dfs - list of other_dfs filtered to sampled customer_ids
    """

    if sample_size is not None:
        num_to_sample = sample_size
    else:
        num_to_sample = int(len(customer_df) * sample_percent)

    sampled_customer_df = customer_df.sample(num_to_sample, random_state=seed)
    sample_customers = sampled_customer_df["customer_id"]
    sampled_other_dfs = [
        other_df[other_df["customer_id"].isin(sample_customers)].copy()
        for other_df in other_dfs
    ]

    return sampled_customer_df, sampled_other_dfs
