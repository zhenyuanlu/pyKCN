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
>>> csv_extractor = CSVExtractor("/path/to/csv/files", ["column1", "column2"], "date_column")
>>> csv_data = csv_extractor.extract_data()
>>> excel_extractor = ExcelExtractor("/path/to/excel/files", ["column1", "column2"], "date_column")
>>> excel_data = excel_extractor.extract_data()

"""

import re
import os
import ast
import pandas as pd


class BaseExtractor:
    """
    Base class for data extraction.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str], date_column: str = None, date_type: str = 'year'):
        """
        Initialize the data directory, columns to extract, and optional date column.
        :param data_dir: The directory where the data files are located.
        :param columns_to_extract: List of column names to extract from the data files.
        :param date_column: Optional; name of the date column in the data files.
        :param data_type: Optional; the type of date values in the date column.
                          Allowed values are 'year', 'numeric', 'strings'. Default is 'year'.
        """
        self.data_dir = data_dir
        self.columns_to_extract = columns_to_extract
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
         safely evaluate a Python literal string and convert it to its corresponding Python data type,
         such as a list, dictionary, tuple, etc., e.g. 'list[str]' -> list[str].
        :param input_str: data from columns
        :return: original python data types
        """
        try:
            return ast.literal_eval(input_str)
        except (ValueError, SyntaxError):
            return input_str

    @staticmethod
    def extract_year(date:str) -> int | None:
        """
        Extract the year from a date string, e.g. 02 Oct 2023 --> 2023.
        :param date: The date string to extract the year from.
        :return: The extracted year as an integer or None.
        """
        year_match = re.search(r'\b\d{4}\b', str(date))
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

    def clean_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the date column in the DataFrame.
        :param df: DataFrame to clean.
        :return: DataFrame with the cleaned data column.
        """
        # If the date column is already of integer type, no processing is needed
        if pd.api.types.is_integer_dtype(df[self.date_column]):
            return df

        # Otherwise, apply the appropriate extraction function
        df[self.date_column] = df[self.date_column].apply(self.date_extractor)

        df = df.dropna(subset = [self.date_column])
        if self.date_type in ['year', 'numeric']:
            df[self.date_column] = df[self.date_column].astype(int)
        return df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the DataFrame according to specified column manipulations.
        :param df: Pandas data frame input to preprocess.
        :return: Preprocessed data frame.
        """
        if self.date_column:
            df = self.clean_date_column(df)
            # df[self.date_column] = pd.to_numeric(df[self.date_column], errors = 'coerce')
            # df = df.dropna(subset = [self.date_column])
            # df[self.date_column] = df[self.date_column].astype(int)

        for column in self.columns_to_extract:
            df[column] = df[column].apply(self.safe_literal_eval)
            df[column] = df[column].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)

        return df

    def extract_data(self) -> pd.DataFrame:
        """
        Base method for data extraction, intended to be overridden by subclasses.
        """
    raise NotImplementedError("This method should be overridden by subclasses.")


class CSVExtractor(BaseExtractor):
    """
    Subclass for extracting data from CSV files.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str], date_column: str = None, date_type: str = 'year'):
        super().__init__(data_dir, columns_to_extract, date_column, date_type)

    def extract_data(self) -> pd.DataFrame:
        """
        Extract and preprocess data from CSV files.
        :return: Extracted pandas dataframe.
        """
        all_data = []
        # Conditionally include date column
        selected_columns = self.columns_to_extract + ([self.date_column] if self.date_column else [])
        for filename in os.listdir(self.data_dir):
            if filename.endswith('csv'):
                file_path = os.path.join(self.data_dir, filename)
                df = pd.read_csv(file_path, usecols=selected_columns, on_bad_lines='skip')
                df = self.preprocess_dataframe(df)
                all_data.append(df)

        return pd.concat(all_data, ignore_index = True)


class ExcelExtractor(BaseExtractor):
    """
    Subclass for extracting data from Excel files.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str], date_columns: str = None, date_type: str = 'year'):
        super().__init__(data_dir, columns_to_extract, date_columns, date_type)

    def extract_data(self) -> pd.DataFrame:
        """
        Extract and preprocess data from Excel files.
        :return: Extracted pandas dataframe.
        """
        all_data = []
        # Conditionally include date column
        selected_columns = self.columns_to_extract + ([self.date_column] if self.date_column else [])
        for filename in os.listdir(self.data_dir):
            if filename.endswith(('.xls', '.xlsx')):
                file_path = os.path.join(self.data_dir, filename)
                df = pd.read_excel(file_path, usecols=selected_columns, header=0)
                df = self.preprocess_dataframe(df)
                all_data.append(df)

        return pd.concat(all_data, ignore_index = True)











