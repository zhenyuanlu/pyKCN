import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def split_into_time_windows(df, split_column, start_value=None, end_value=None, window_size=None, num_windows=None):
    """
    Split the DataFrame into time windows based on a specified column, with optional
    start and end values, and either a specific window size or number of windows.

    :param df: pandas DataFrame containing the data to split.
    :param split_column: Name of the column to use for splitting into windows.
    :param start_value: The starting value for the first window (inclusive).
    :param end_value: The ending value for the last window (inclusive).
    :param window_size: The size of each window (number of groups per window).
    :param num_windows: The total number of windows to split the data into.
    :return: Dictionary with window details as keys and DataFrames as values.
    """
    if (window_size is None) == (num_windows is None):
        raise ValueError("Please specify either window_size or num_windows, not both.")

    # Convert start_value and end_value to appropriate types based on the data type of split_column
    if start_value is not None:
        if pd.api.types.is_datetime64_any_dtype(df[split_column]):
            start_value = pd.to_datetime(start_value)
        elif pd.api.types.is_integer_dtype(df[split_column]):
            start_value = int(start_value)

    if end_value is not None:
        if pd.api.types.is_datetime64_any_dtype(df[split_column]):
            end_value = pd.to_datetime(end_value)
        elif pd.api.types.is_integer_dtype(df[split_column]):
            end_value = int(end_value)

    # Filter the DataFrame for the specified range and remove rows with zero or NaN values
    df_filtered = df[~df[split_column].isin([0, np.nan])]
    if start_value is not None:
        df_filtered = df_filtered[df_filtered[split_column] >= start_value]
    if end_value is not None:
        df_filtered = df_filtered[df_filtered[split_column] <= end_value]

    if df_filtered.empty:
        logging.warning("No data available after applying start_value "
                        "and end_value filters. Returning an empty dictionary.")
        return {}

    # Sort the DataFrame once for window processing
    sorted_df = df_filtered.sort_values(by = split_column)

    overall_start = sorted_df[split_column].min()
    overall_end = sorted_df[split_column].max()

    # Add an "overall" entry to the windows dictionary to represent the entire dataset
    windows = {('overall', (overall_start, overall_end)): sorted_df}

    # Determine the window size if num_windows is specified
    if num_windows:
        total_values = sorted_df[split_column].nunique()
        window_size = max(total_values // num_windows, 1)  # Ensure window size is at least 1

    unique_values = sorted_df[split_column].unique()

    # Generate the windows based on window size
    for i in range(0, len(unique_values), window_size):
        window_start = unique_values[i]
        window_end = unique_values[min(i + window_size - 1, len(unique_values) - 1)]
        window_df = sorted_df[(sorted_df[split_column] >= window_start) & (sorted_df[split_column] <= window_end)]

        window_key = (window_start, window_end)
        if pd.api.types.is_datetime64_any_dtype(sorted_df[split_column]):
            window_key = tuple(map(lambda x: x.strftime('%Y-%m-%d'), window_key))

        windows[window_key] = window_df

    return windows

