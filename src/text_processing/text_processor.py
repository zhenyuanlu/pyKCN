import re
import string
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from collections import defaultdict

from .base_processor import BaseProcessor


class TokenProcessor(BaseProcessor):

    TEXT_PROCESSING_STEPS = [
        {"description": "Removing Angle Bracket Pattern...", "function": "remove_angle_brackets",
         "args": {}},
        {"description": "Splitting Text...", "function": "split_by_delimiter", "args": {}},
        {"description": "Lowering Cases...", "function": "to_lowercase", "args": {}},
        {"description": "Unicode Normalization...", "function": "unicode_normalize", "args": {}},
        {"description": "Tokenizing...", "function": "tokenize_string", "args": {}},
        {"description": "Removing Non ASCII...", "function": "remove_non_ascii", "args": {}},
        {"description": "Splitting Terms by Hyphen...", "function": "handle_hyphenated_terms", "args": {}},
        # {"description": "Stripping Whitespace...", "function": "strip_whitespace", "args": {}},
        # {"description": "Removing Numbers...", "function": "remove_numbers",
        #  "args": {"pattern_type": "all"}},
        # {"description": "Filtering by Length...", "function": "filter_by_length", "args": {}},
        {"description": "Removing Punctuation...", "function": "remove_punctuation",
         "args": {"punctuation_type": "default"}},
    ]
    # 'machine-learning method; lstm, deep learning'
    # ['machine-learning method', 'lstm', 'deep learning']
    # [['machine-learning', 'method'], [lstm], ['deep', ' learning']]
    # [['machine', 'learning', 'method'], ['lstm'], ['deep', 'learning']]
    # then save this original words to the vocabulary , and another copy for stemming
    # rejoin each term back to single string for both original and stemmed

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
        self.pipeline = self.TEXT_PROCESSING_STEPS
        self.new_col_name = new_col_name

    def execute_processor(self) -> pd.DataFrame:

        new_col_name = self.new_col_name
        self.combine_columns(self.columns_to_process, new_col_name = new_col_name)
        # Get the column data
        if new_col_name in self.dataframe.columns:
            for step in self.pipeline:
                description = step["description"]
                function = getattr(self, step["function"])
                args = step["args"]
                self.dataframe[new_col_name] = self.apply_with_progress(self.dataframe[new_col_name],
                                                                        function,
                                                                        description,
                                                                        **args)


            # Store original tokens for updating vocabulary and dictionary
            # Stem tokens
            # Final cleanup - remove empty tokens and None values from both stemmed tokens and original tokens
            # Rejoin terms - rejoin both stemmed tokens and original tokens at the top level
            # Update vocabulary and dictionary


            self.dataframe.drop(columns = ['col_1', 'col_2'], inplace = True)
        return self.dataframe

