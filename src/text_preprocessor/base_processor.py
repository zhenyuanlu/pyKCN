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
                 columns_to_process: list[str],
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
        self.handle_nan()

    def handle_nan(self):
        """Handles NaN values based on a strategy."""
        if self.fill_na is None:
            self.dataframe.dropna(subset = self.columns_to_process, how = 'all', inplace = True)
            self.dataframe.dropna(subset = self.columns_to_deduplicate, how = 'all', inplace = True)
        else:
            self.dataframe.fillna(self.fill_na, inplace = True)
        pass

    def validate_dataframe(self):
        """Validates the DataFrame structure and types."""
        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError("dataframe should be a Pandas DataFrame")
        if self.dataframe.empty:
            raise ValueError("The DataFrame should not be empty.")
        if not all(col in self.dataframe.columns for col in self.columns_to_process):
            raise ValueError("All columns_to_process must exist in the DataFrame.")
        if not all(col in self.dataframe.columns for col in self.columns_to_deduplicate):
            raise ValueError("All columns_to_deduplicate must exist in the DataFrame.")
        # We may add additional validations.

    def apply_custom_operations(self, operations: list):
        """Allows users to apply their own list of operations."""
        for operation in operations:
            self.dataframe[self.columns_to_process] = self.dataframe[self.columns_to_process].applymap(operation)

    def combine_columns(self,
                        columns_to_combine: list,
                        new_col_name = 'target_col') -> None:
        """
        Combine the specified columns into a new column in the dataframe.
        :param self:
        :param columns_to_combine:
        :param new_col_name:
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

    def process_deduplication(self) -> pd.DataFrame:
        """Overrides the execute_processing method from BaseTextProcessor."""
        self.combine_columns(self.columns_to_deduplicate, new_col_name = 'temp_col')
        if 'temp_col' in self.dataframe.columns:
            # Before tokenizing, set the description for tqdm's progress bar
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'],
                                                                  self.tokenize_string,
                                                                  'Tokenizing...')
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'],
                                                                  self.remove_punctuation,
                                                                  'Removing Punctuation...',
                                                                  punctuation_type = 'default')
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'],
                                                                  self.remove_numbers,
                                                                  'Removing Numbers...',
                                                                  pattern_type = 'all')
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'],
                                                                  self.to_lowercase,
                                                                  'Lowering Cases...')
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'],
                                                                  self.unicode_normalize,
                                                                  'Unicode Normalization...')

            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(' '.join)
            self.dataframe = self.remove_duplicates(self.dataframe, 'temp_col', self.deduplication_threshold)

            # Drop the temporary column after deduplication
            # self.dataframe.drop(columns = ['temp_col'], inplace = True)
        return self.dataframe

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

    def remove_duplicates(self, df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
        """
        Deduplicate a DataFrame based on exact matches or string similarity.

        :param df: DataFrame to deduplicate.
        :param column: Column to check for duplicates.
        :param threshold: If 1.0, perform exact deduplication. Otherwise, use string similarity.
        :return: Deduplicated DataFrame.
        """
        threshold_percentage = int(threshold * 100)
        if threshold_percentage == 100:
            return df.drop_duplicates(subset = [column])
        else:
            # Deduplicate based on string similarity
            return self.deduplicate_based_on_similarity(df, column, threshold_percentage)

    @staticmethod
    def is_similar(string1: str, string2: str, threshold_percentage: int) -> bool:
        """
        Check if two strings are similar based on a given threshold.

        :param string1: First string.
        :param string2: Second string.
        :param threshold_percentage: Similarity threshold.
        :return: Boolean indicating if strings are similar.
        """
        # Levenshtein Distance
        return fuzz.ratio(string1, string2) >= threshold_percentage

    def deduplicate_based_on_similarity(self, df: pd.DataFrame, column: str, threshold_percentage: int) -> pd.DataFrame:
        """
        Deduplicate a DataFrame based on string similarity.

        :param df: Input DataFrame.
        :param column: Column to check for duplicates.
        :param threshold_percentage: Similarity threshold.
        :return: Deduplicated DataFrame.
        """
        unique_rows = []
        checked_strings = []

        for _, row in df.iterrows():
            current_string = row[column]
            if any(self.is_similar(current_string, checked_string, threshold_percentage) for checked_string in
                   checked_strings):
                continue
            checked_strings.append(current_string)
            unique_rows.append(row)

        return pd.DataFrame(unique_rows)

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

    def final_cleanup(self, tokens):
        """Final cleanup operations: stripping and removing None values."""
        return [token.strip() for token in tokens if token and token.strip()]

    @staticmethod
    def rejoin_terms(self, tokens):
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

    def update_vocabulary_and_dictionary(self, original_tokens, stemmed_tokens):
        for orig, stem in zip(original_tokens, stemmed_tokens):
            self.vocabulary.add(orig)
            self.stem_to_original[stem].add(orig)

    @staticmethod
    def apply_with_progress(df: pd.DataFrame, function, description, *args, **kwargs) -> pd.DataFrame:
        tqdm.pandas(desc = description)
        df = df.progress_apply(lambda x: function(x, *args, **kwargs))
        return df
