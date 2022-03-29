import os
import sys
import pickle as pkl


def pickle_variables(variable_names, variables_dict, delete_variables=False):
    """
    saves to current directory
    if delete_variables=True, will also delete variables

    pickle variables with format <variable_name>.pkl
    """
    for variable_name in variable_names:
        with open(f"{variable_name}.pkl", "wb") as f:
            pkl.dump(variables_dict[variable_name], f)
        if delete_variables:
            del variables_dict[variable_name]


def unpickle_variables(variable_names, variables_dict, delete_files=False):
    """
    load pickled objects from current directory
    if delete_objs=True, will also delete pickled files

    will look for pickled files with format <variable_name>.pkl
    """

    for variable_name in variable_names:
        file_name = f"{variable_name}.pkl"
        with open(file_name, "rb") as f:
            variables_dict[variable_name] = pkl.load(f)
        if delete_files:
            os.remove(file_name)


def report_memory_used(variable_names, variables_dict):
    """
    reports on memory used by each of the variables, in mb
    also reports on total memory used
    """
    variables = [variables_dict[variable_name] for variable_name in variable_names]
    memories = [round(sys.getsizeof(variable) / 1000000) for variable in variables]
    total_memory = sum(memories)

    memory_messages = [
        f"{variable_names[i]}: {memories[i]:,}mb" for i in range(len(variable_names))
    ]
    memory_message = ", ".join(memory_messages)
    print(memory_message)
    print(f"total memory used: {total_memory:,}mb")


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
