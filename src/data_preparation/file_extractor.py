"""
FileExtractor Module
================

"""


import os
import pandas as pd
from .base_extractor import BaseExtractor


class FileExtractor(BaseExtractor):
    """
    Subclass for extracting data from CSV/Excel files.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str] | dict[str, list[str]],
                 date_column: list | str = None, date_type: str = 'year'):
        super().__init__(data_dir, columns_to_extract, date_column, date_type)

    def get_columns_by_filename(self, filename: str) -> set:
        """
        Get columns to extract based on the filename.

        :param filename: The name of the file.
        :return: Set of columns to extract.
        """
        # Consider various ways a user might refer to a filename
        possible_keys = {filename, os.path.splitext(filename)[0], filename.split('.')[0]}
        for key in possible_keys:
            if key in self.columns_to_extract:
                return set(self.columns_to_extract[key])
        return set()

    def get_columns_by_file_extension(self, file_extension: str) -> set:
        """
        Get columns to extract based on the file extension.

        :param file_extension: The file extension.
        :return: Set of columns to extract.
        """
        # Consider various ways a user might refer to a file extension
        possible_keys = {file_extension, file_extension.lstrip('.'), '.' + file_extension.lstrip('.')}
        for key in possible_keys:
            if key in self.columns_to_extract:
                return set(self.columns_to_extract[key])
        return set()

    def determine_columns(self, filename: str) -> set:
        """
        Determine which columns to extract based on filename or file extension.

        :param filename: The name of the file.
        :return: Set of columns to extract.
        """
        file_extension = os.path.splitext(filename)[1]

        if isinstance(self.columns_to_extract, dict):
            return (self.get_columns_by_filename(filename)
                    or self.get_columns_by_file_extension(file_extension))
        elif isinstance(self.columns_to_extract, list):
            return set(self.columns_to_extract)
        else:
            raise ValueError("columns_to_extract must be either a list or a dictionary.")

    @staticmethod
    def read_file(filepath: str, potential_columns: set) -> pd.DataFrame | None:
        """
        Read the file and return a DataFrame with the required columns.

        :param filepath: The full path of the file to read.
        :param potential_columns: Set of potential columns to extract.
        :return: DataFrame with the extracted data or None if an error occurs.
        """
        try:
            preview_df = pd.read_csv(filepath, nrows = 1) \
                if filepath.endswith('.csv') \
                else pd.read_excel(filepath, nrows = 1)
            available_columns = set(preview_df.columns)
            columns_to_use = list(available_columns.intersection(potential_columns))

            if not columns_to_use:
                print(f"No matching columns found in {filepath}. Skipping.")
                return None

            if filepath.endswith('.csv'):
                return pd.read_csv(filepath, usecols = columns_to_use, on_bad_lines='skip')
            elif filepath.endswith(('.xls', '.xlsx')):
                return pd.read_excel(filepath, usecols = columns_to_use, header=0)
            else:
                return None  # Unsupported file type
        except Exception as e:
            print(f"An error occurred while reading {filepath}: {e}")
            return None

    def extract_data(self) -> pd.DataFrame:
        """
        Extract and preprocess data from CSV/Excel files.
        :return: Extracted pandas dataframe.
        """
        all_data = []
        for filename in os.listdir(self.data_dir):
            potential_columns = self.determine_columns(filename)
            if potential_columns:
                filepath = os.path.join(self.data_dir, filename)
                df = self.read_file(filepath, potential_columns)
                if df is not None:
                    # Include date column if specified
                    selected_columns = potential_columns.union(
                        {self.date_column}) if self.date_column else potential_columns
                    df = df[selected_columns]

                    # Call to preprocess_dataframe from BaseExtractor
                    df = self.preprocess_dataframe(df)

                    all_data.append(df)
        return pd.concat(all_data, ignore_index = True)

