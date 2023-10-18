"""
TextProcessor
=============


"""

import os
from datetime import datetime
import pandas as pd
from .base_processor import BaseProcessor
from ..utils.utils import is_package_installed


# Steps
# 'machine-learning method; lstm, deep learning'
# ['machine-learning method', 'lstm', 'deep learning']
# [['machine-learning', 'method'], [lstm], ['deep', ' learning']]
# [['machine', 'learning', 'method'], ['lstm'], ['deep', 'learning']]

# then save this original words to the vocabulary , and another copy for stemming
# rejoin each term back to single string for both original and stemmed


class TextProcessor(BaseProcessor):
    DEFAULT_PRIMARY_PIPELINE = {
        'default': [
            {"description": "Removing Angle Bracket Pattern...", "function": "remove_angle_brackets",
             "args": {}},
            # 'machine-learning method; lstm, deep learning'
            # ['machine-learning method', 'lstm', 'deep learning']
            {"description": "Splitting Text...", "function": "split_by_delimiter", "args": {}},
            {"description": "Lowering Cases...", "function": "to_lowercase", "args": {}},
            {"description": "Unicode Normalization...", "function": "unicode_normalize", "args": {}},
            # [['machine-learning', 'method'], [lstm], ['deep', ' learning']]
            {"description": "Tokenizing...", "function": "tokenize_string", "args": {}},
            {"description": "Removing Non ASCII...", "function": "remove_non_ascii", "args": {}},
            # [['machine', 'learning', 'method'], ['lstm'], ['deep', 'learning']]
            {"description": "Splitting Terms by Hyphen...", "function": "handle_hyphenated_terms", "args": {}},
            {"description": "Removing Stopwords...", "function": "remove_stopwords", "args": {}},
            {"description": "Stripping Whitespace...", "function": "strip_whitespace", "args": {}},

            # TODO - Create an argument for custom pattern type.
            {"description": "Removing Numbers...", "function": "remove_numbers",
             "args": {"pattern_type": "all"}},

            {"description": "Removing Punctuation...", "function": "remove_punctuation",
             "args": {"punctuation_type": "default"}},
        ]
    }

    DEFAULT_STEMMING_PIPELINE = {
        'original_data': [
            {"description": "Rejoining Original Terms...", "function": "rejoin_terms", "args": {}},
            {"description": "Filtering by Length...", "function": "filter_by_length", "args": {}},
            {"description": "Cleaning Original Terms...", "function": "cleanup", "args": {}},
        ],
        'stemmed_data': [
            {"description": "Stemming...", "function": "stem_tokens", "args": {}},
            {"description": "Rejoining Stemmed Terms...", "function": "rejoin_terms", "args": {}},
            {"description": "Filtering by Length...", "function": "filter_by_length", "args": {}},
            {"description": "Cleaning Stemmed Terms...", "function": "cleanup", "args": {}},
        ]
    }

    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns_to_process: list[str],
                 date_column: str = 'date_col',
                 deduplication_threshold: float = 1.0,
                 new_col_name = 'text_col',
                 word_len_threshold: int = 2,
                 primary_pipeline: list[dict] = None,
                 stemming_pipeline: list[dict] = None,
                 cache_location: str = None,
                 fill_na: str = None,
                 cache_format: str = 'csv'):
        super().__init__(dataframe = dataframe,
                         columns_to_process = columns_to_process,
                         date_column = date_column,
                         deduplication_threshold = deduplication_threshold,
                         word_len_threshold = word_len_threshold,
                         fill_na = fill_na)
        self.handle_nan(mode_type = 'processing')
        self.new_col_name = new_col_name

        self.PRIMARY_PIPELINE = primary_pipeline or self.DEFAULT_PRIMARY_PIPELINE
        self.STEMMING_PIPELINE = stemming_pipeline or self.DEFAULT_STEMMING_PIPELINE

        self.primary_pipeline_data = None  # to store the processed data from the primary pipeline
        self.stemming_pipeline_data = None  # to store the processed data from the stemming pipeline

        self.cache_location = cache_location
        self.cache_format = cache_format.lower()

    def execute_processor(self, run_primary = True, run_stemming = True) -> pd.DataFrame:

        new_col_name = self.new_col_name
        self.combine_columns(self.columns_to_process, new_col_name = new_col_name)

        if self.cache_location:
            # Check for cached primary data
            if run_primary:
                cached_data = self.load_cached_data('primary')
                if cached_data is not None:
                    self.dataframe = cached_data
                    run_primary = False  # Skip primary pipeline if cached data is loaded

            # Check for cached stemming data
            if run_stemming:
                cached_data = self.load_cached_data('stemming')
                if cached_data is not None:
                    self.dataframe = cached_data
                    run_stemming = False

        if run_primary:
            self.execute_primary_pipeline()
            if self.cache_location:
                self.save_cached_data('primary')

        if run_stemming:
            self.execute_stemming_pipeline()
            if self.cache_location:
                self.save_cached_data('stemming')

        return self.dataframe

    def execute_primary_pipeline(self):
        """Execute the primary processing pipeline."""
        new_col_name = self.new_col_name
        if new_col_name in self.dataframe.columns:
            for _, pipeline in self.PRIMARY_PIPELINE.items():
                self.dataframe[new_col_name] = self._process_pipeline(self.dataframe,
                                                                      new_col_name, pipeline)

        self.primary_pipeline_data = self.dataframe.copy()

    def execute_stemming_pipeline(self):
        """Execute the stemming and secondary processing pipeline."""
        new_col_name = self.new_col_name
        # Prepare for the next pipeline
        self.dataframe['original_data'] = self.dataframe[new_col_name].copy()
        self.dataframe['stemmed_data'] = self.dataframe[new_col_name].copy()

        for column, pipeline in self.STEMMING_PIPELINE.items():
            self.dataframe[column] = self._process_pipeline(self.dataframe, column, pipeline)

        # Drop the columns that are not needed
        cols_to_keep = self.columns_to_process + ['original_data', 'stemmed_data']
        cols_to_drop = set(self.dataframe.columns) - set(cols_to_keep)
        self.dataframe.drop(columns = cols_to_drop, inplace = True)

        self.stemming_pipeline_data = self.dataframe.copy()

    # ==================================
    # Helper Method
    # ==================================
    def _process_pipeline(self, df: pd.DataFrame, column_name: str, pipeline: list[dict]) -> pd.DataFrame:
        """
        Process a column in the DataFrame based on the provided pipeline.

        :param df: DataFrame to process.
        :param column_name: Column to process.
        :param pipeline: Pipeline from the class attribute.
        :return: Processed DataFrame Column.
        """
        for step in pipeline:
            description = step["description"]
            function = getattr(self, step["function"])
            args = step["args"]
            df[column_name] = self.apply_with_progress(df[column_name], function, description, **args)
        return df[column_name]

    def save_cached_data(self, pipeline_type: str) -> None:
        """Save the current DataFrame to the cache."""
        cache_file_path = self.get_cache_file_path(pipeline_type)

        if self.cache_format == 'csv':
            self.dataframe.to_csv(cache_file_path, index = False)
        elif self.cache_format == 'parquet':
            self._handle_parquet_format()
            self.dataframe.to_parquet(cache_file_path)
        else:
            raise ValueError(f"Unsupported cache format: {self.cache_format}")

    def load_cached_data(self, pipeline_type: str) -> pd.DataFrame | None:
        """
        Load the DataFrame from the cache if it exists.

        :param pipeline_type:
        :return:
        """
        file_extension = 'parquet' if self.cache_format == 'parquet' else 'csv'
        # List all cache files for the specific pipeline
        cache_files = [f for f in os.listdir(self.cache_location)
                       if f.startswith(f"cache_{pipeline_type}_") and f.endswith(f".{file_extension}")]

        # If no cache files, return None
        if not cache_files:
            return None

        # Sort cache files by timestamp and pick the latest
        latest_cache_file = sorted(cache_files, key = self.extract_timestamp)[-1]
        cache_file_path = os.path.join(self.cache_location, latest_cache_file)

        if self.cache_format == "csv":
            return pd.read_csv(cache_file_path)
        elif self.cache_format == "parquet":
            self._handle_parquet_format()
            return pd.read_parquet(cache_file_path)
        return None

    # TODO - Move extract_timestamp to utils
    @staticmethod
    def extract_timestamp(filename: str) -> datetime:
        # Strip file extension and extract the timestamp
        timestamp_str = "_".join(filename.split("_")[-2:]).split('.')[0]
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

    def get_cache_file_path(self, pipeline_type: str) -> str:
        """Get the cache file path based on the current DataFrame's content."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = self.cache_format
        return os.path.join(self.cache_location, f"cache_{pipeline_type}_{timestamp}.{file_extension}")

    def get_primary_pipeline_data(self, use_cache: bool = False) -> pd.DataFrame:
        """Return the processed data from the primary pipeline."""
        if self.primary_pipeline_data is not None:
            return self.primary_pipeline_data.copy()
        else:
            raise ValueError(
                "Primary pipeline data not available. "
                "Ensure the primary pipeline has been executed or valid cache is available.")

    def get_stemming_pipeline_data(self, use_cache: bool = False) -> pd.DataFrame:
        """Return the processed data from the stemming pipeline."""
        if self.stemming_pipeline_data is not None:
            return self.stemming_pipeline_data.copy()
        else:
            raise ValueError(
                "Stemming pipeline data not available. "
                "Ensure the stemming pipeline has been executed or valid cache is available.")

    @staticmethod
    def _handle_parquet_format() -> pd.DataFrame | None:
        """
        Handle operations for the 'parquet' cache format.

        :return: None
        """
        custom_error_msg = ("To use the 'parquet' format, you need to install the 'pyarrow' package. "
                            "You can install it using pip or conda.")
        if not is_package_installed('pyarrow', custom_error_msg):
            return
