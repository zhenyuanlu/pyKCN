import re
import string
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize, combining
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
                 custom_delimiter: str = r'\s*[;,()/]\s*|\s-\s',
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

        self.custom_punctuation_pattern = re.compile(r'[%s]' % re.escape(custom_punctuation))
        self.default_punctuation_pattern = re.compile(r'[%s]' % re.escape(default_punctuation))
        self.non_ascii_pattern  = re.compile(r'[^\x00-\x7F]+')
        self.angle_brackets_pattern = re.compile(r'<[^>]+>.*?</[^>]+>')

        # Targets numbers that are standalone or sandwiched between spaces.
        # "Hello 456 World" -> "Hello World"
        self.pattern_numbers_with_spaces = re.compile(r'^\d+\s|\s\d+\s|\s\d+$')
        # Matches numbers that are standalone. "12345", "9", "000001"
        self.pattern_standalone_numbers = re.compile(r'^\d+$')
        # Strips all numbers within tokens, even those embedded within text. "abc123def" -> "abcdef"
        self.pattern_all_numerics = re.compile(r'\d+')

        self.vocabulary = set()
        self.stem_to_original = defaultdict(set)
        self.validate_dataframe()

        self.STRING_TO_LIST_METHODS = {}
        # self.STRING_TO_LIST_METHODS = {self.split_by_delimiter, self.tokenize_string, self.handle_hyphenated_terms}
        self.LIST_TO_STRING_METHODS = {self._handle_hyphens_in_single_token}

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
        return self._text_input_handler(text, self._split_single_string)

    def tokenize_string(self, text):
        return self._text_input_handler(text, self._tokenize_single_string)

    # def handle_hyphenated_terms(self, tokens):
    #     return self._text_input_handler(tokens, self._handle_hyphens_in_single_token)

    # def handle_hyphenated_terms(self, tokens):
    #     """Process hyphenated words."""
    #     processed_data = self._text_input_handler(tokens, self._handle_hyphens_in_single_token)
    #
    #     if isinstance(tokens, str):
    #         return processed_data
    #
    #     # Flatten the inner list structure resulting from hyphen splits
    #     if isinstance(tokens, list) and all(isinstance(item, str) for item in tokens):
    #         return [term for sublist in processed_data for term in sublist]
    #
    #     # For a list of lists of strings
    #     flattened_data = []
    #     for sublist in processed_data:
    #         flattened_sublist = [term for subsublist in sublist for term in subsublist]
    #         flattened_data.append(flattened_sublist)
    #     return flattened_data

    def handle_hyphenated_terms(self, tokens):
        return self._text_input_handler(tokens, self._handle_hyphens_in_single_token)

    def filter_by_length(self, tokens):
        return self._text_input_handler(tokens, self._filter_by_length_single_token)

    # pattern_numbers_with_spaces/pattern_standalone_numbers/pattern_embedded_numbers
    def remove_numbers(self, tokens: list[str], pattern_type: str = 'all') -> list[str]:
        """
        Remove numbers from the tokens based on the provided patterns.
        :param tokens: List of tokens
        :param pattern_type: The pattern type for number removal
        :return: Tokens with numbers removed
        """
        # Ensure the pattern type is valid
        valid_pattern_types = {"spaces", "standalone", "all"}
        if pattern_type not in valid_pattern_types:
            raise ValueError(f"Invalid pattern_type. Choose from {', '.join(valid_pattern_types)}.")

        # Use the _text_input_handler to process the tokens
        return self._text_input_handler(tokens, self._remove_numbers_from_single_token, pattern_type)

    def remove_punctuation(self, data, punctuation_type = 'default'):
        return self._text_input_handler(data, self._remove_punctuation_single_token,
                                        punctuation_type = punctuation_type)

    def stem_tokens(self, tokens):
        return self._text_input_handler(tokens, self._stem_single_token)

    @staticmethod
    def rejoin_terms(tokens):
        if isinstance(tokens, str):
            return tokens  # It's already a string.
        elif all(isinstance(item, str) for item in tokens):
            return ' '.join(tokens)
        elif all(isinstance(sublist, list) for sublist in tokens) and all(
                isinstance(item, str) for sublist in tokens for item in sublist):
            return [' '.join(sublist) for sublist in tokens]

    def to_lowercase(self, text: str) -> str:
        """Converts all strings to lowercase."""
        return self._text_input_handler(text, self._to_lowercase_single_string)

    def unicode_normalize(self, text: str) -> str:
        return self._text_input_handler(text, self._unicode_normalize_single_string)

    def remove_non_ascii(self, tokens):
        return self._text_input_handler(tokens, self._remove_non_ascii_single_string)

    def remove_angle_brackets(self, tokens):
        return self._text_input_handler(tokens, self._remove_angle_brackets_single_string)

    def strip_whitespace(self, tokens):
        return self._text_input_handler(tokens, self._strip_whitespace_from_single_token)

    def remove_stopwords(self, tokens):
        return self._text_input_handler(tokens, self._remove_stopwords_from_token)

    def final_cleanup(self, tokens: str) -> str:
        return self._text_input_handler(tokens, self._cleanup_single_token)

    def cleanup(self, tokens: list[str]) -> list[str]:
        return [cleaned for cleaned in (self._cleanup_single_token(token) for token in tokens) if cleaned]

    @staticmethod
    def apply_with_progress(data: pd.DataFrame | str | list, function, description, *args,
                            **kwargs) -> pd.DataFrame | list:
        if isinstance(data, pd.Series):
            tqdm.pandas(desc = description)
            return data.progress_apply(lambda x: function(x, *args, **kwargs))

        elif isinstance(data, pd.DataFrame):
            tqdm.pandas(desc = description)
            return data.progress_applymap(lambda x: function(x, *args, **kwargs))

        else:
            raise ValueError("Invalid data type. `apply_with_progress` expects a pandas Series or DataFrame.")

# ==================================
# Text input handler
# ==================================

    def _text_input_handler(self, input_data, processing_function, *args, **kwargs):
        # Apply transformations
        transformed_data = self._apply_transformation(input_data, processing_function, *args, **kwargs)

        # Adjust data structure
        if processing_function in self.STRING_TO_LIST_METHODS:
            return self._wrap_strings_in_list(transformed_data)
        elif processing_function in self.LIST_TO_STRING_METHODS:
            return self._remove_extra_nesting(transformed_data)
        else:
            return transformed_data

    def _apply_transformation(self, input_data, processing_function, *args, **kwargs):
        if isinstance(input_data, str):
            processed_data = processing_function(input_data, *args, **kwargs)
            return processed_data if processed_data else ''
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            processed_data  = [processing_function(item, *args, **kwargs) for item in input_data]
            return [item for item in processed_data if item]
        elif isinstance(input_data, list) and all(isinstance(sublist, list) for sublist in input_data):
            processed_data = [self._apply_transformation(sublist, processing_function, *args, **kwargs)
                              for sublist in input_data]
            return [sublist for sublist in processed_data if sublist]
        else:
            raise ValueError("Invalid input type.")

    def _wrap_strings_in_list(self, data):
        """Increase the level of nested lists by wrapping strings in a list."""
        if isinstance(data, str):
            return [data]
        elif isinstance(data, list) and all(isinstance(item, str) for item in data):
            return [[item] for item in data]
        elif isinstance(data, list) and all(isinstance(sublist, list) for sublist in data):
            return [self._wrap_strings_in_list(sublist) for sublist in data]
        else:
            raise ValueError("Invalid data type for _increase_list_level.")

    @staticmethod
    def _flatten_one_level(data):
        """Flatten the list by one level."""
        if isinstance(data, list) and all(isinstance(item, list) for item in data):
            return [item for sublist in data for item in sublist]
        else:
            return data

    @staticmethod
    def _remove_extra_nesting(nested_list: list) -> list:
        """
        Flattens one level of nesting in a list, but retains the top-level structure.
        """
        flattened = []
        for sublist in nested_list:
            new_sublist = []
            for item in sublist:
                if isinstance(item, list):
                    new_sublist.extend(item)
                else:
                    new_sublist.append(item)
            flattened.append(new_sublist)
        return flattened


# ==================================
# Helper Methods
# ==================================
    # Helper methods
    def _split_single_string(self, text: str) -> list[str]:
        return re.split(self.custom_delimiter, text)

    @staticmethod
    def _tokenize_single_string(text: str) -> list[str]:
        return word_tokenize(text)

    @staticmethod
    def _handle_hyphens_in_single_token(token: str) -> list[str]:
        """Split the token by hyphen and return the resulting parts."""
        return token.split('-')

    @staticmethod
    def _handle_hyphens_in_list(self, tokens: list) -> list:
        """
        Handle hyphens in a list of tokens. If we use remove_extra_nesting above,
        then we can remove this method.

        :param tokens: List of tokens
        :return: List of tokens with hyphens handled
        """
        processed_tokens = []
        for token in tokens:
            if '-' in token:
                processed_tokens.extend(token.split('-'))
            else:
                processed_tokens.append(token)
        return processed_tokens

    def _filter_by_length_single_token(self, token: str) -> str:
        return token if len(token) > self.word_len_threshold else ''

    def _remove_numbers_from_single_token(self, token: str, pattern_type: str) -> str:
        """
        Helper function to remove numbers from a single token based on the provided patterns.
        :param token: The input token
        :param pattern_type: The pattern type for number removal
        :return: Token with numbers removed
        """
        valid_pattern_types = {
            "spaces": self.pattern_numbers_with_spaces,
            "standalone": self.pattern_standalone_numbers,
            "all": self.pattern_all_numerics
        }

        pattern = valid_pattern_types.get(pattern_type)

        if pattern_type == "spaces" and not pattern.search(token):
            return token
        elif pattern_type == "standalone" and not pattern.fullmatch(token):
            return token
        elif pattern_type == "all":
            cleaned_token = re.sub(pattern, ' ', token)
            if cleaned_token.strip():
                return re.sub(r'\s+', ' ', cleaned_token)
        return ''

    def _remove_punctuation_single_token(self, token: str, punctuation_type: str = 'default') -> str:
        """Removes punctuation from a single token."""
        if punctuation_type == 'custom':
            cleaned_token = self.custom_punctuation_pattern.sub('', token)
        else:  # Default
            cleaned_token = self.default_punctuation_pattern.sub('', token)

        return cleaned_token

    def _stem_single_token(self, token: str) -> str:
        return self.stemmer.stem(token) if self.stemmer else token

    @staticmethod
    def _to_lowercase_single_string(text: str) -> str:
        return text.lower()

    @staticmethod
    def _unicode_normalize_single_string(text: str) -> str:
        nfkd_form = normalize('NFKD', text)
        return nfkd_form.translate(dict.fromkeys([ord(c) for c in nfkd_form if combining(c)]))

    @staticmethod
    def _remove_angle_brackets_single_string(text: str) -> str:
        return re.sub(r'<[^>]+>.*?</[^>]+>', '', text)

    @staticmethod
    def _remove_non_ascii_single_string(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def _strip_whitespace_from_single_token(token: str) -> str:
        return token.strip()

    def _remove_stopwords_from_token(self, token: str) -> str:
        return token if token not in self.stop_words else ''

    @staticmethod
    def _cleanup_single_token(token: str) -> str | None:
        if token is None:
            return None
        cleaned_token = token.strip()
        return cleaned_token if cleaned_token else None
