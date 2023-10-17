import copy
import re
import string
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from collections import defaultdict

from .base_processor import BaseProcessor
from ..utils.utils import save_data_from_prep, load_data_from_prep

# 'machine-learning method; lstm, deep learning'
# ['machine-learning method', 'lstm', 'deep learning']
# [['machine-learning', 'method'], [lstm], ['deep', ' learning']]
# [['machine', 'learning', 'method'], ['lstm'], ['deep', 'learning']]

# then save this original words to the vocabulary , and another copy for stemming
# rejoin each term back to single string for both original and stemmed


class TextProcessor(BaseProcessor):
    FIRST_PIPELINE = {
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

    SECOND_PIPELINE = {
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
                 fill_na: str = None):
        super().__init__(dataframe = dataframe,
                         columns_to_process = columns_to_process,
                         date_column = date_column,
                         deduplication_threshold = deduplication_threshold,
                         word_len_threshold = word_len_threshold,
                         fill_na = fill_na)
        self.handle_nan(mode_type = 'processing')
        self.new_col_name = new_col_name

    def execute_processor(self) -> pd.DataFrame:

        new_col_name = self.new_col_name
        self.combine_columns(self.columns_to_process, new_col_name = new_col_name)
        # Get the column data
        if new_col_name in self.dataframe.columns:
            # The first pipeline
            for _, pipeline in self.FIRST_PIPELINE.items():
                self.dataframe[new_col_name] = self._process_pipeline(self.dataframe,
                                                                      new_col_name, pipeline)

            # After the first pipeline, create the new columns 'original_data' and 'stemmed_data'
            # This is just an example; replace with your actual logic to generate these columns.
            self.dataframe['original_data'] = self.dataframe[new_col_name].copy()
            self.dataframe['stemmed_data'] = self.dataframe[new_col_name].copy()

            # Process the second pipeline
            for column, pipeline in self.SECOND_PIPELINE.items():
                self.dataframe[column] = self._process_pipeline(self.dataframe, column, pipeline)

            # Drop the columns that are not needed

            cols_to_keep = self.columns_to_process + ['original_data', 'stemmed_data']
            cols_to_drop = set(self.dataframe.columns) - set(cols_to_keep)
            self.dataframe.drop(columns = cols_to_drop, inplace = True)
        return self.dataframe

    # ==================================
    # Helper Method
    # ==================================
    def _process_pipeline(self, df: pd.DataFrame, column_name: str, pipeline: list[dict]) -> pd.DataFrame:
        """
        Process a column in the DataFrame based on the provided pipeline.

        :param df:
        :param column_name:
        :param pipeline:
        :return:
        """
        for step in pipeline:
            description = step["description"]
            function = getattr(self, step["function"])
            args = step["args"]
            df[column_name] = self.apply_with_progress(df[column_name], function, description, **args)
        return df[column_name]
