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
    def __init__(self,
                 data_mapping: dict = None,
                 new_column_names: list[str] = None,
                 data_dir: str = None,
                 corpus_columns: list[str] = None,
                 date_column: str = None,
                 date_type: str = 'year'):
        """
        Initialize the BaseExtractor class.
        Parameters:
            data_mapping (dict, optional): Mapping for file inputs.
            data_dir (str, optional): Parent directory for file inputs.
            corpus_columns (list[str], optional): Columns for in-memory DataFrame.
            date_column (str, optional): Date columns for in-memory DataFrame.
            date_type (str, optional): Type of date, default is 'year'.

        Raises:
            ValueError: If both sets of parameters or incomplete sets are provided.
            TypeError: If the wrong type of data is provided.
            KeyError: If expected keys are not found in data_mapping.
        """
        self.data_mapping = data_mapping
        self.new_column_names = new_column_names
        self.data_dir = data_dir
        self.corpus_columns = corpus_columns
        self.date_column = date_column
        self.date_type = date_type
        self.validate_inputs()

    def validate_inputs(self):
        """
        Validate the initial inputs.
        """
        self._validate_exclusive_parameters()
        self._validate_data_mapping_and_dir()
        self._validate_corpus_and_date_columns()
        self._validate_types()

    def _validate_exclusive_parameters(self):
        """
        Ensure either file input or in-memory DataFrame parameters are provided, but not both.
        """
        if (self.data_mapping or self.data_dir) and (self.corpus_columns or self.date_column):
            raise ValueError(
                "Provide either file input parameters (data_mapping and data_dir) "
                "or in-memory DataFrame parameters (corpus_column and date_column), but not both."
            )

    def _validate_data_mapping_and_dir(self):
        """
        Validate data_mapping and data_dir.
        """
        if self.data_mapping and not self.data_dir:
            raise ValueError("If data_mapping is provided, data_dir must also be provided.")

    def _validate_corpus_and_date_columns(self):
        """
        Validate corpus_column and date_column.
        """
        if self.corpus_columns and not self.date_column:
            raise ValueError("If corpus_column is provided, date_column must also be provided.")

    def _validate_types(self):
        """
        Validate the types of provided parameters.
        """
        if self.data_mapping:
            if not isinstance(self.data_mapping, dict):
                raise TypeError("data_mapping must be a dictionary.")
            for folder, config in self.data_mapping.items():
                if not isinstance(folder, str) or not isinstance(config, dict):
                    raise TypeError("data_mapping keys must be strings and values must be dictionaries.")
                if 'corpus_columns' not in config or 'date_column' not in config:
                    raise KeyError(
                        "Each folder configuration in data_mapping must include 'corpus_columns' and 'date_column'.")

        if self.data_dir and not isinstance(self.data_dir, str):
            raise TypeError("data_dir must be a string.")

        if self.corpus_columns:
            if not all(isinstance(item, str) for item in self.corpus_columns):
                raise TypeError("All items in corpus_column must be strings.")

        if self.date_column and not isinstance(self.date_column, str):
            raise TypeError("date_column must be a string.")

        if self.date_type and not isinstance(self.date_type, str):
            raise TypeError("date_type must be a string.")



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
    def extract_year_group(date: str) -> int | None:
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
            'year': self.extract_year_group,
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
            # df.loc[:, date_column] = pd.to_numeric(df.loc[:, date_column], errors = 'coerce')
            df.loc[:, date_column] = df[date_column].astype(int)
        return df

    def load_data(self) -> pd.DataFrame | None:
        """
        Base method for loading data, intended to be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def extract_data(self) -> pd.DataFrame:
        """
        Extract and preprocess data.
        :return: Extracted pandas dataframe.
        """
        df, corpus_columns, date_column = self.load_data()

        if date_column:
            df[date_column] = pd.to_numeric(df[date_column], errors = 'coerce')
            df = df.dropna(subset = [date_column])
            df.loc[:, date_column] = df[date_column].astype(int)
            df = self.clean_date_column(df, date_column)
        #     # df[self.date_column] = pd.to_numeric(df[self.date_column], errors = 'coerce')
        #     df = df.dropna(subset = [self.date_column])
        #     # df[self.date_column] = df[self.date_column].astype(int)


        for column in corpus_columns:
            df.loc[:, column] = df[column].apply(self.safe_literal_eval)
            df.loc[:, column] = df[column].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)
        #
        #             # Call to preprocess_dataframe from BaseExtractor
        return df


