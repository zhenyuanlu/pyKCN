"""
Extractor Module
================
This module provides classes for extracting data from various file formats such as CSV and Excel.
The BaseExtractor class serves as the superclass that contains common functionalities, including
methods for preprocessing DataFrames.

Subclasses like CSVExtractor and ExcelExtractor extend BaseExtractor to provide specific functionalities
for extracting data from CSV and Excel files, respectively.

Classes:
--------
- BaseExtractor: Base class for data extraction.
- CSVExtractor: Subclass for extracting data from CSV files.
- ExcelExtractor: Subclass for extracting data from Excel files.

Example:
--------
# >>> csv_extractor = CSVExtractor("/path/to/csv/files", ["column1", "column2"], "date_column")
# >>> csv_data = csv_extractor.extract_data()
# >>> excel_extractor = ExcelExtractor("/path/to/excel/files", ["column1", "column2"], "date_column")
# >>> excel_data = excel_extractor.extract_data()

"""

import re
import os
import ast
import pandas as pd


class BaseExtractor:
    """
    Base class for data extraction.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str] | dict[str, list[str]],
                 date_column: list | str = None, date_type: str = 'year'):
        """
        Initialize the data directory, columns to extract, and optional date column.
        :param data_dir: The directory where the data files are located.
        :param columns_to_extract: Either a list of columns to extract from all files, or a dictionary mapping
                                   filenames or file extensions to lists of columns to extract.
        :param date_column: Optional; name of the date column in the data files.
        :param data_type: The type of date values in the date column.
                          Allowed values are 'year', 'numeric', 'strings'. Default is 'year'.
        """
        self.data_dir = data_dir
        self.columns_to_extract = columns_to_extract # Can be either a list or a dictionary
        self.date_column = date_column
        self.date_type = date_type
        self.validate_inputs()

    def validate_inputs(self):
        """
        Validate the initial inputs.
        """
        if not os.path.exists(self.data_dir):
            raise ValueError("The specified data directory does not exist.")
        if not self.columns_to_extract:
            raise ValueError("columns_to_extract cannot be empty.")

    @staticmethod
    def safe_literal_eval(input_str: str) -> any:
        """
        Safely evaluate a Python literal string and convert it to its corresponding Python data type,
        such as a list, dictionary, tuple, etc., e.g. 'list[str]' -> list[str].
        :param input_str: data from columns
        :return: original python data types
        """
        try:
            return ast.literal_eval(input_str)
        except (ValueError, SyntaxError):
            return input_str

    @staticmethod
    def extract_year(date: str) -> int | None:
        """
        Extract the year from a date string, e.g. 02 Oct 2023 --> 2023.
        :param date: The date string to extract the year from.
        :return: The extracted year as an integer or None.
        """
        year_match = re.search(r'\d{4}', str(date))
        if year_match:
            return int(year_match.group(0))
        return None

    @staticmethod
    def extract_numeric_group(date: str) -> int | None:
        """
        Extract the numeric group from a date string.
        :param date: The date string to extract the numeric group from.
        :return: The extracted numeric group as an integer or None.
        """
        try:
            return int(date)
        except ValueError:
            return None

    @staticmethod
    def extract_string_group(date: str) -> str:
        """
        Extract the string group from a date string.
        :param date: The date string to extract the string group from.
        :return: The extracted string group.
        """
        return str(date)

    def date_extractor(self, date: str) -> any:
        """
        General date extraction method based on date_type.
        :param date: Date column.
        :return: Any date groups.
        """
        extraction_methods = {
            'year': self.extract_year,
            'numeric': self.extract_numeric_group,
            'string': self.extract_string_group
        }

        extraction_function = extraction_methods.get(self.date_type)

        if not extraction_function:
            raise ValueError(f'Invalid date_type: {self.date_type}')

        return extraction_function(date)

    def clean_date_column(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Clean the date column in the DataFrame.
        :param df: DataFrame to clean.
        :param date_column: Name of the date column to clean.
        :return: DataFrame with the cleaned data column.
        """
        # If the date column is already of integer type, no processing is needed
        if pd.api.types.is_integer_dtype(df[date_column]):
            return df

        # Otherwise, apply the appropriate extraction function
        df[date_column] = df[date_column].apply(self.date_extractor)

        df = df.dropna(subset = [date_column])
        if self.date_type in ['year', 'numeric']:
            df[date_column] = df[date_column].astype(int)
        return df

    def determine_date_column(self, df: pd.DataFrame) -> str | None:
        """
        Determine the actual date column to use based on DataFrame columns.

        :param df: DataFrame to examine.
        :return: The name of the date column to use, or None if not found.
        """
        if isinstance(self.date_column, list):
            for col in self.date_column:
                if col in df.columns:
                    return col
            print('No matching date column found: list')
            return None # No matching date column found
        elif isinstance(self.date_column, str):
            if self.date_column in df.columns:
                return self.date_column
            else:
                print('No matching date column found: str')
                return None
        else:
            print('No date column selected: None')
            return None

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame according to specified column manipulations.
        :param df: Pandas data frame input to preprocess.
        :return: Preprocessed data frame.
        """
        actual_date_column = self.determine_date_column(df)
        if actual_date_column:
            df = self.clean_date_column(df, actual_date_column)
            # df[self.date_column] = pd.to_numeric(df[self.date_column], errors = 'coerce')
            # df = df.dropna(subset = [self.date_column])
            # df[self.date_column] = df[self.date_column].astype(int)

        for column in self.columns_to_extract:
            df[column] = df[column].apply(self.safe_literal_eval)
            df[column] = df[column].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)

        return df

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
        :return: List of columns to extract.
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
        :return: List of columns to extract.
        """
        file_extension = os.path.splitext(filename)[1]

        if isinstance(self.columns_to_extract, dict):
            return (self.get_columns_by_filename(filename)
                    or self.get_columns_by_file_extension(file_extension))
        elif isinstance(self.columns_to_extract, list):
            return set(self.columns_to_extract)
        else:
            raise ValueError("columns_to_extract must be either a list or a dictionary.")

    def get_data_sources(self) -> any:
        """
        Retrieve data sources, intended to be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load_data(self, source, potential_columns: set) -> pd.DataFrame | None:
        """
        Base method for loading data, intended to be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def extract_data(self) -> pd.DataFrame:
        """
        Extract and preprocess data.
        :return: Extracted pandas dataframe.
        """
        all_data = []
        for source in self.get_data_sources():
            potential_columns = self.determine_columns(source)
            if potential_columns:
                df = self.load_data(source, potential_columns)
                # print(df)
                if df is not None:
                    # Include date column if specified
                    selected_columns = potential_columns.union(
                        set(self.date_column)) if self.date_column else potential_columns
                    df = df[selected_columns]

                    # Call to preprocess_dataframe from BaseExtractor
                    df = self.preprocess_dataframe(df)
                    all_data.append(df)
        return pd.concat(all_data, ignore_index = True)
