import re
import string
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from unicodedata import normalize
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class BaseProcessor:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns_to_process: list[str] = None,
                 columns_to_deduplicate: list[str] = None,
                 date_column: str = 'date_col',
                 custom_delimiter: str = r',; - /()',
                 custom_punctuation: str = r'!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~',
                 default_punctuation: str = string.punctuation,
                 deduplication_threshold: float = 1.0,
                 word_len_threshold: int = 2,
                 fill_na: str = None):
        self.dataframe = dataframe.copy()
        self.columns_to_process = columns_to_process
        self.columns_to_deduplicate = columns_to_deduplicate
        self.date_column = date_column
        self.custom_delimiter = custom_delimiter
        self.deduplication_threshold = deduplication_threshold
        self.word_len_threshold = word_len_threshold
        self.fill_na = fill_na
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.custom_punctuation_pattern = re.compile('[%s]' % re.escape(custom_punctuation))
        self.default_punctuation_pattern = re.compile('[%s]' % re.escape(default_punctuation))

        # Targets numbers that are standalone or sandwiched between spaces.
        # "Hello 456 World" -> "Hello World"
        self.pattern_numbers_with_spaces = re.compile(r'^\d+\s|\s\d+\s|\s\d+$')
        # Matches numbers that are standalone. "12345", "9", "000001"
        self.pattern_standalone_numbers = re.compile(r'^\d+$')
        # Strips all numbers within tokens, even those embedded within text. "abc123def" -> "abcdef"
        self.pattern_embedded_numbers = re.compile(r'\d+')
        self.pattern_all_numerics = re.compile(r'^\d+$|\d+')

        self.vocabulary = set()
        self.stem_to_original = defaultdict(set)
        self.validate_dataframe()

    def handle_nan(self, mode_type = 'deduplication'):
        """
        Handles NaN values based on a process.

        :param mode_type: Specifies the mode for NaN handling. Accepted values are 'deduplication' and 'processing'.
                          Based on this mode, the appropriate columns are targeted for NaN handling.
                          Default is 'deduplication'.
        :return: None. The dataframe is modified in-place.
        """
        # Columns to target based on mode_type
        target_columns = self.columns_to_deduplicate if mode_type == 'deduplication' else self.columns_to_process

        # Apply the NaN handling strategy
        if self.fill_na is None:
            self.dataframe.dropna(subset = target_columns, how = 'all', inplace = True)
        else:
            self.dataframe[target_columns].fillna(self.fill_na, inplace = True)

    def validate_dataframe(self):
        """Validates the DataFrame structure and types."""
        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError("dataframe should be a Pandas DataFrame")
        if self.dataframe.empty:
            raise ValueError("The DataFrame should not be empty.")
        # if not all(col in self.dataframe.columns for col in self.columns_to_process):
        #     raise ValueError("All columns_to_process must exist in the DataFrame.")
        # if not all(col in self.dataframe.columns for col in self.columns_to_deduplicate):
        #     raise ValueError("All columns_to_deduplicate must exist in the DataFrame.")
        # We may add additional validations.

    def apply_custom_operations(self, operations: list):
        """Allows users to apply their own list of operations."""
        for operation in operations:
            self.dataframe[self.columns_to_process] = self.dataframe[self.columns_to_process].applymap(operation)

    def combine_columns(self,
                        columns_to_combine: list[str],
                        new_col_name = 'target_col') -> None:
        """
        Combine the specified columns into a new column in the dataframe.

        :param columns_to_combine: The target columns for combination.
        :param new_col_name: The new column name with default value, 'target_col'.
        :return:
        """

        if not columns_to_combine:
            raise ValueError("No columns provided for combination.")

        for col in columns_to_combine:
            if col not in self.dataframe.columns:
                raise ValueError(f"Column '{col}' not found in the dataframe.")

        # Check if the new column name already exists
        original_new_col_name = new_col_name
        counter = 1
        while new_col_name in self.dataframe.columns:
            new_col_name = f"{original_new_col_name}_{counter}"
            counter += 1
        if new_col_name != original_new_col_name:
            print(f"Warning: Column name '{original_new_col_name}' already exists. Using '{new_col_name}' instead.")

        # If only one column is specified, just copy that column
        if len(columns_to_combine) == 1:
            self.dataframe[new_col_name] = self.dataframe[columns_to_combine[0]]
        else:
            self.dataframe[new_col_name] = self.dataframe[columns_to_combine].apply(
                lambda row: ','.join(row.dropna().values.astype(str)), axis = 1
            )
        # return self.dataframe

    def execute_processor(self) -> pd.DataFrame:
        """
        Base method for executing the processor, intended to be overridden by subclasses.

        :return: Processed pandas dataframe
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def split_by_delimiter(self, text):
        return re.split(self.custom_delimiter, text)

    @staticmethod
    def tokenize_string(text: str) -> list[str]:
        """
        Tokenizes the text in DataFrame.
        :param text:
        :return:
        """
        return word_tokenize(text)

    @staticmethod
    def handle_hyphenated_terms(self, tokens):
        """Process hyphenated words."""
        return [word for token in tokens for word in re.split(r'-', token)]

    @staticmethod
    def strip_whitespace(self, tokens):
        """Strip leading/trailing whitespace from each token."""
        return [token.strip() for token in tokens]

    def filter_by_length(self, tokens):
        """Remove tokens based on length."""
        return [token for token in tokens if len(token) > self.word_len_threshold]

    def process_corpus_columns(self):
        pass

    # pattern_numbers_with_spaces/pattern_standalone_numbers/pattern_embedded_numbers
    def remove_numbers(self, tokens: list[str], pattern_type = 'standalone') -> list[str]:
        """
        Remove numbers based on the provided patterns.
        :param tokens:
        :param pattern_type:
        :return:
        """
        valid_pattern_types = ["spaces", "standalone", "embedded", "all"]
        if pattern_type not in valid_pattern_types:
            raise ValueError(f"Invalid pattern_type. Choose from {', '.join(valid_pattern_types)}.")

        if pattern_type == "spaces":
            return [token for token in tokens if not self.pattern_numbers_with_spaces.match(token)]
        elif pattern_type == "standalone":
            return [token for token in tokens if not self.pattern_standalone_numbers.match(token)]
        elif pattern_type == "embedded":
            return [re.sub(self.pattern_embedded_numbers, '', token) for token in tokens]
        elif pattern_type == "all":
            processed_tokens = []
            for token in tokens:
                # Remove numbers using the combined pattern
                token_without_numbers = re.sub(self.pattern_all_numerics, '', token)
                if token_without_numbers:  # Check to ensure token is not empty
                    processed_tokens.append(token_without_numbers.strip())
            return processed_tokens

    def remove_punctuation(self, tokens: list[str], punctuation_type = 'default') -> list[str]:
        """
        Removes punctuation from the text in DataFrame column/s,
        :param tokens:
        :param punctuation_type:
        :return:
        """
        if punctuation_type == 'custom':
            return [self.custom_punctuation_pattern.sub("", token) for token in tokens]
        else:  # Default
            return [self.default_punctuation_pattern.sub("", token) for token in tokens]

    def stem_tokens(self, tokens):
        """Apply stemming."""
        if self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens

    @staticmethod
    def final_cleanup(tokens):
        """Final cleanup operations: stripping and removing None values."""
        return [token.strip() for token in tokens if token and token.strip()]

    @staticmethod
    def rejoin_terms(tokens):
        """Join cleaned tokens back into a single string."""
        return ' '.join(tokens)

    @staticmethod
    def to_lowercase(tokens: list[str]) -> list[str]:
        """Converts all strings to lowercase."""
        return [token.lower() for token in tokens]

    @staticmethod
    def unicode_normalize(tokens: list[str]) -> list[str]:
        """Perform Unicode normalization."""
        return [normalize('NFKD', token) for token in tokens]


    @staticmethod
    def apply_with_progress(df: pd.DataFrame, function, description, *args, **kwargs) -> pd.DataFrame:
        tqdm.pandas(desc = description)
        df = df.progress_apply(lambda x: function(x, *args, **kwargs))
        return df
