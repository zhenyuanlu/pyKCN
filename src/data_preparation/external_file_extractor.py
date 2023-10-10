"""
ExternalExtractor Module
========================
This module provides the ExternalExtractor class for extracting data from various external sources like databases and APIs.
The ExternalExtractor class is a specialized subclass of the BaseExtractor, extending its capabilities to handle data extraction
from a variety of external data sources.

The BaseExtractor class serves as the superclass that contains common functionalities, including methods for
preprocessing DataFrames.

The ExternalExtractor subclass leverages these common functionalities and adds specific methods to interact with external data sources.

Classes:
--------
- BaseExtractor: Base class for data extraction (imported from another module).
- ExternalExtractor: Subclass for extracting data from various external sources like databases and APIs.

Example:
--------
# Define the directory and mapping for data extraction
data_dir = "/path/to/data/files"
data_mapping = {
    'source1': {'corpus_columns': ['Title', 'Keywords'], 'date_column': 'Published Date'},
    'source2': {'corpus_columns': ['Document Title', 'Tags'], 'date_column': 'Publication Year'}
}

# Initialize and use ExternalFileExtractor
external_file_extractor = ExternalFileExtractor(data_mapping=data_mapping, data_dir=data_dir, date_type='year')
external_data = external_file_extractor.extract_data()
"""

import os
import logging
from typing import Tuple, List

import pandas as pd
from pandas import DataFrame

from .base_extractor import BaseExtractor

# File Constants
CSV_EXTENSIONS = ('.csv',)
EXCEL_EXTENSIONS = ('.xls', '.xlsx')
READ_ERROR = 'An error occurred while reading file {} : {}'


class ExternalFileExtractor(BaseExtractor):
    """
    Subclass for extracting data from CSV/Excel files.

    Attributes:
        data_mapping (dict): Mapping between folders and their column configurations.
        data_dir (str): Directory where the data folders are located.
        new_column_names (dict, optional): Custom names for columns.
        date_type (str, optional): Type of date information ('year', 'numeric', or 'string').
        logger (logging.Logger): Logger instance for error logging.
    """

    def __init__(self,
                 data_mapping: dict[str] | None = None,
                 data_dir: str = None,
                 new_column_names: list[str] | None = None,
                 date_type: str = 'year'):
        """
        Initialize the ExternalFileExtractor class.

        :param data_mapping: Mapping between folders and their column configurations.
        :param data_dir: Directory where the data folders are located.
        :param new_column_names: Custom names for columns.
        :param date_type: Type of date information ('year', 'numeric', or 'string').

        :raise LoggingError: If an error occurs while logging.
        """
        super().__init__(data_mapping = data_mapping,
                         data_dir = data_dir,
                         new_column_names = new_column_names,
                         date_type = date_type)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> tuple[pd.DataFrame, list[str], str]:
        """
        Load data from all specified folders and concatenate into a single DataFrame.

        :return: Final concatenated DataFrame, final corpus columns, and final date column.
        """
        all_data = []
        final_corpus_columns = None
        final_date_column = None
        for folder, config in self.data_mapping.items():
            folder_data, new_column_names, final_corpus_columns, final_date_column = (
                self._load_data_from_folder(folder, config))
            if folder_data is not None:
                all_data.append(folder_data)

        final_df = pd.concat(all_data, ignore_index = True)
        return final_df, final_corpus_columns, final_date_column

    def _load_data_from_folder(self, folder: str, config: dict) -> tuple[DataFrame | None, list[str], list[str], str]:
        """
        Load data from a specific folder based on its configuration.

        :param folder: Name of the folder.
        :param config: Configuration for the folder's data, a list of corpus column names,
                        and a string of date column name
        :return: Concatenated DataFrame from the folder, new column names, final corpus columns, and final date column.
        """
        corpus_columns = config.get('corpus_columns')
        date_column = config.get('date_column')
        folder_path = os.path.join(self.data_dir, folder)
        target_columns = corpus_columns + [date_column]
        new_column_names, final_corpus_columns, final_date_column = (
            self._get_new_column_names(corpus_columns, date_column))

        data_frames = []
        for file in os.listdir(folder_path):
            file_data = self._load_data_from_file(folder_path, file, target_columns)
            if file_data is not None:
                file_data.rename(columns = new_column_names, inplace = True)
                data_frames.append(file_data)

        return (pd.concat(data_frames, ignore_index = True) if data_frames else None,
                new_column_names, final_corpus_columns, final_date_column)

    def _load_data_from_file(self,
                             folder_path: str,
                             file: str,
                             target_columns: list[str]) -> pd.DataFrame | None:
        """
        Load data from a specific file, CSV or Excel.

        :param folder_path: Path to the folder containing the file.
        :param file: File name with file extension.
        :param target_columns: List of target column names.
        :return: DataFrame containing data from the file, or None.
        """
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(CSV_EXTENSIONS):
                return pd.read_csv(file_path, usecols = target_columns, on_bad_lines = 'skip')
            elif file.endswith(EXCEL_EXTENSIONS):
                return pd.read_excel(file_path, usecols = target_columns, header = 0)
        except Exception as e:
            self._log_error(READ_ERROR.format(file_path, e))
        return None

    def _get_new_column_names(self,
                              corpus_columns: list[str],
                              date_column: str) -> tuple[list[str], list[str], str]:
        """
        Generate new column names based on provided or default values.

        :param corpus_columns: List of corpus columns.
        :param date_column: Name of the date column.
        :return: New column names, final corpus columns, and final date column.
        """
        if not self.new_column_names:
            new_column_names = {col: f"col_{i + 1}" for i, col in enumerate(corpus_columns)}
            new_column_names[date_column] = 'date_col'
        else:
            new_column_names = self.new_column_names

        final_corpus_columns, final_date_column = self._get_final_column_names(corpus_columns,
                                                                               date_column,
                                                                               new_column_names)
        return new_column_names, final_corpus_columns, final_date_column

    @staticmethod
    def _get_final_column_names(corpus_columns: list[str],
                                date_column: str,
                                new_column_names: dict) -> tuple[list[str], str]:
        """
        Finalize the corpus and date column names.

        :param corpus_columns: List of corpus columns.
        :param date_column: Name of the date column.
        :param new_column_names: New column names.
        :return: Final corpus columns and final date column.
        """
        # Getting new corpus column names
        final_corpus_columns = [new_column_names[col] for col in corpus_columns]
        # Getting the new date column name
        final_date_column = new_column_names[date_column]
        return final_corpus_columns, final_date_column

    def _log_error(self, error_message: str) -> None:
        """
        Log an error message.

        :param error_message: The error message to log.
        :return: None
        """
        self.logger.error(error_message)
