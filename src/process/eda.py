from typing import Tuple

import numpy as np
import pandas as pd


def get_correlated_features(
    df: pd.DataFrame,
    corr_thr: float,
    method: str = "spearman",
    sampled: bool = True,
) -> Tuple[list, pd.DataFrame]:
    """Calculates correlation for a given dataframe.

    Args:
        df (pd.DataFrame): input data.
        corr_thr (float): Correlation threshold.
        method (str, optional): Correlation method.

    Returns:
        Tuple[list, pd.DataFrame]: list with correlated columns.
    """
    if sampled:
        n_rows = 200_000 if len(df) > 200_000 else len(df)
        df = df.sample(n=n_rows, random_state=42)
    corr_matrix = df.corr(method=method).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    cols_to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]

    return cols_to_drop, corr_matrix


def get_constant_value_cols(df: pd.DataFrame) -> list:
    """Get columns with constant (same) values."""
    return [e for e in df.columns if df[e].nunique() == 1]


def get_ncategories_categorical_cols(df, top: int = 10):
    """
    Find categorical columns in a Pandas DataFrame with more than a specified number of unique values.

    Parameters:
    df (Pandas DataFrame): The input DataFrame.
    top (int): The maximum number of categorical columns to return. Default is 10.

    Returns:
    A list of tuples, where each tuple contains the name of a categorical column with the highest number of unique values and the number of unique values.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    result = {}
    for col in cat_cols:
        unique_vals = df[col].nunique()
        result[col] = unique_vals
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)[:top]
    return sorted_result


def details(df):
    sum_null_values = df.isnull().sum()
    percent_null_values = 100 * (sum_null_values / len(df))
    data_type = df.dtypes
    unique_values = df.nunique()

    table = pd.concat(
        [sum_null_values, percent_null_values, data_type, unique_values], axis=1
    )
    table_col = table.rename(
        columns={
            0: "Missing Values",
            1: "% of Total Missing Values",
            2: "Data_Type",
            3: "Unique values",
        }
    )
    return table_col
