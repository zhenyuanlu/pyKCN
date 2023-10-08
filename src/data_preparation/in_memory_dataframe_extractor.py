"""
InMemoryDataFrameExtractor Module
=================================
This module provides the InMemoryDataFrameExtractor class, which is a specialized subclass of the BaseExtractor.
It is designed for extracting data from in-memory pandas DataFrames.

The BaseExtractor class serves as the superclass, containing common functionalities for data extraction and preprocessing.

Classes:
--------
- BaseExtractor: Base class for data extraction (imported from another module).
- InMemoryDataFrameExtractor: Subclass for extracting data from in-memory pandas DataFrames.

Example:
--------
# Assuming df is a pandas DataFrame object with relevant columns
data_mapping = {
    'corpus_columns': ['Title', 'Keywords'],
    'date_column': 'Published Date'
}

# Initialize and use InMemoryDataFrameExtractor
in_memory_data_extractor = InMemoryDataFrameExtractor(data_frame=df, data_mapping=data_mapping, date_type='year')
in_memory_data = in_memory_data_extractor.extract_data()
"""

import logging
import pandas as pd
from .base_extractor import BaseExtractor

READ_ERROR = 'An error occurred during data loading: {}'


class InMemoryDataFrameExtractor(BaseExtractor):
    """
    Subclass for extracting data from local data frame.
    """

    def __init__(self,
                 data_frame: pd.DataFrame,
                 new_column_names: list[str] = None,
                 corpus_columns: list[str] = None,
                 date_column: str = None,
                 date_type: str = 'year'):
        super().__init__(data_frame = data_frame,
                         new_column_names = new_column_names,
                         corpus_columns = corpus_columns,
                         date_column = date_column,
                         date_type = date_type, )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> tuple[pd.DataFrame, list[str], str]:
        """
        Load and preprocess data from the in-memory DataFrame based on the column mapping.

        :return: Final concatenated DataFrame, final corpus columns, and final date column.
        """
        try:
            target_columns = self.corpus_columns + [self.date_column]

            # Extract relevant columns
            final_df = self.data_frame[target_columns]
            final_corpus_columns = self.corpus_columns
            final_date_column = self.date_column

            if self.new_column_names:
                new_column_mapping = dict(zip(target_columns, self.new_column_names))
                final_df.rename(columns = new_column_mapping, inplace = True)
                final_corpus_columns = [new_column_mapping[col] for col in self.corpus_columns]
                final_date_column = new_column_mapping[self.date_column]
            return final_df, final_corpus_columns, final_date_column
        except Exception as e:
            self._log_error(READ_ERROR.format(e))
            return pd.DataFrame(), [], ""

    def _log_error(self, error_message: str) -> None:
        """
        Log an error message.

        :param error_message: The error message to log.
        :return: None
        """
        self.logger.error(error_message)
