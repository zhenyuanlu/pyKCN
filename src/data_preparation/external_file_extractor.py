"""

================

"""


import os
import pandas as pd
from .base_extractor import BaseExtractor


class ExternalFileExtractor(BaseExtractor):
    """
    Subclass for extracting data from CSV/Excel files.
    """
    def __init__(self, data_dir: str, columns_to_extract: list[str] | dict[str, list[str]],
                 date_column: list | str = None, date_type: str = 'year'):
        super().__init__(data_dir, columns_to_extract, date_column, date_type)

    def get_data_sources(self) -> list[str]:
        """
        Retrieve filenames as data sources from the directory.
        """
        return os.listdir(self.data_dir)

    def load_data(self, source, potential_columns: set) -> pd.DataFrame | None:
        """
        Read the file and return a DataFrame with the required columns.

        :param source: The data source
        :param potential_columns: Set of potential columns to extract.
        :return: DataFrame with the extracted data or None if an error occurs.
        """
        try:
            preview_df = pd.read_csv(source, nrows = 1) \
                if source.endswith('.csv') \
                else pd.read_excel(source, nrows = 1)
            available_columns = set(preview_df.columns)
            columns_to_use = list(available_columns.intersection(potential_columns))

            if not columns_to_use:
                print(f"No matching columns found in {source}. Skipping.")
                return None

            if source.endswith('.csv'):
                return pd.read_csv(source, usecols = columns_to_use, on_bad_lines='skip')
            elif source.endswith(('.xls', '.xlsx')):
                return pd.read_excel(source, usecols = columns_to_use, header=0)
            else:
                return None  # Unsupported file type
        except Exception as e:
            print(f"An error occurred while reading {source}: {e}")
            return None



