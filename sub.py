import pandas as pd
from typing import Union, List
import copy


def remove_duplicates_list(orig_list: list) -> list:
    """
    remove duplicates from python list, while retaining order
    """

    unique_list = copy.deepcopy(orig_list)

    for idx in range(len(unique_list), 1, -1):
        if unique_list[idx - 1] in unique_list[: idx - 1]:
            unique_list.pop(idx - 1)

    return unique_list


def create_sub(
    customer_ids: Union[List[str], pd.Series],
    predictions: Union[dict, pd.Series],
    default_predictions: List[str] = [],
) -> pd.Series:
    """
    given predictions for some customers,
    generates a pd.Series for all customers, in the competition submission format.

    predictions are limited to first 12
    customer ids without predictions provided will be blank
    customer ids with less that 12 predictions (or none provided)
        will be augmented with default_predictions, if provided
        default_predictions will be added in the order they are present in the list

    args:
    customer_ids: all the article ids we need to submit for
    predictions: pd.Series where index=customer_id and values=list of predictions
    default_predictions: predictions to augment customer_id predictions with, until 12
    """

    # original predictions or empty list
    sub = pd.DataFrame({"customer_id": customer_ids})
    sub["prediction"] = sub["customer_id"].map(predictions)
    sub["prediction"] = sub["prediction"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    sub["prediction"] = sub["prediction"].apply(lambda x: x + default_predictions)
    sub["prediction"] = sub["prediction"].apply(remove_duplicates_list)
    sub["prediction"] = sub["prediction"].apply(lambda x: x[:12])
    sub["prediction"] = sub["prediction"].apply(lambda x: " ".join(x))

    return sub
