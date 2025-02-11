import pandas as pd

def sharedColumns(*dfs):
    """
    :param dfs:
    :return: set of shared columns
    """
    if not dfs:
        return set()
    elif len(dfs) == 1:
        return set(dfs[0].columns)
    else:
        current_columns = set(dfs[0].columns)
        remaining_common_columns = sharedColumns(*dfs[1:])
        return current_columns & remaining_common_columns

def match_data_types(df1, df2):
    """
    Compare data types of two datasets and change types in df2 to match df1.

    Parameters:
    df1 (pd.DataFrame): First dataset with desired data types.
    df2 (pd.DataFrame): Second dataset to be adjusted.

    Returns:
    pd.DataFrame: Adjusted second dataset with matching data types.
    """
    type_compare = pd.DataFrame({"df1_dtypes": df1.dtypes, "df2_dtypes": df2.dtypes})

    for column in df1.columns:
        if column in df2.columns and df1[column].dtype != df2[column].dtype:
            print(f"Changing {column}: {df2[column].dtype} to {df1[column].dtype}")
            df2[column] = df2[column].astype(df1[column].dtype)

    return df2
