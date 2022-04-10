from typing import Union, List
import copy
import pickle as pkl

import pandas as pd
import cudf


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
    customer_ids: Union[List[str], cudf.Series],
    predictions: Union[dict, cudf.Series],
    index_to_id_dict_path: str = None,
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
    index_to_id_dict_path: mapping to get from number index to the customer_id for submission
    """

    # can't support cudf for map
    predictions = predictions.to_pandas().apply(list)
    customer_ids = customer_ids.to_pandas()

    # original predictions or empty list
    sub = pd.DataFrame({"customer_id": customer_ids})
    sub["prediction"] = sub["customer_id"].map(predictions)
    sub["prediction"] = sub["prediction"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    sub["prediction"] = sub["prediction"].apply(lambda x: x + default_predictions)
    sub["prediction"] = sub["prediction"].apply(remove_duplicates_list)
    sub["prediction"] = sub["prediction"].apply(lambda x: x[:12])
    sub["prediction"] = sub["prediction"].apply(
        lambda x: " ".join(["0" + str(article_id) for article_id in x])
    )

    if index_to_id_dict_path is not None:
        index_to_id_dict = pkl.load(open(index_to_id_dict_path, "rb"))

        index_to_id_dict = index_to_id_dict.to_pandas()

        sub["customer_id"] = sub["customer_id"].map(index_to_id_dict)

    return sub


# not sure if we'll need this, so not updating to support cudf
# instead commenting out for deletion soon

# def combine_subs(subs: list):
#     """
#     combine a list of subs
#     - concatenate predictions together for each customer id
#     - deduplicate
#     - take first 12
#     """
#     # this is what we'll return
#     combining_subs = subs[0].copy()

#     # first predictions
#     combining_preds = combining_subs["prediction"]
#     combining_preds = combining_preds.apply(lambda x: x.split(" "))

#     # loop/add other ones
#     for next_sub in subs[1:]:
#         next_pred = next_sub["prediction"]
#         combining_preds = combining_preds + next_pred.apply(lambda x: x.split(" "))

#     # back into submission form
#     combining_preds = combining_preds.apply(remove_duplicates_list)
#     combining_preds = combining_preds.apply(lambda x: x[:12])
#     combining_preds = combining_preds.apply(lambda x: " ".join(x))

#     combining_subs["prediction"] = combining_preds

#     return combining_subs
