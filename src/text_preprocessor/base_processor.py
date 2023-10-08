import string
import re
import pandas as pd
from unicodedata import normalize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from fuzzywuzzy import fuzz

#
# class ProcessorConfig:
#     def __init__(self,
#                  custom_punctuation = r'!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~',
#                  default_punctuation = string.punctuation):
#         self.custom_punctuation = custom_punctuation
#         self.default_punctuation = default_punctuation


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
        self.custom_punctuation = custom_punctuation
        self.default_punctuation = default_punctuation
        self.deduplication_threshold = deduplication_threshold
        self.word_len_threshold = word_len_threshold
        self.fill_na = fill_na
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        self.default_punctuation_pattern = re.compile('[%s]' % re.escape(self.default_punctuation))
        self.custom_punctuation_pattern = re.compile('[%s]' % re.escape(self.custom_punctuation))

        self.pure_num_pattern = re.compile('^\d+$')
        self.validate_dataframe()

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

    def handle_nan(self):
        """Handles NaN values based on a strategy."""
        if self.fill_na is None:
            self.dataframe.dropna(subset = self.columns_to_process, how = 'all', inplace = True)
        else:
            self.dataframe.fillna(self.fill_na, inplace = True)
        pass

    def apply_custom_operations(self, operations: list):
        """Allows users to apply their own list of operations."""
        for operation in operations:
            self.dataframe[self.columns_to_process] = self.dataframe[self.columns_to_process].applymap(operation)

    def create_temp_col(self):
        if self.columns_to_deduplicate:
            self.dataframe['temp_col'] = self.dataframe[self.columns_to_deduplicate].apply(
                lambda row: ','.join(row.dropna().values.astype(str)), axis = 1
            )
        else:
            print("Skipping deduplication step as no columns_to_deduplicate provided.")

    def process_deduplication(self) -> pd.DataFrame:
        """Overrides the execute_processing method from BaseTextProcessor."""
        self.create_temp_col()
        if 'temp_col' in self.dataframe.columns:
            # Before tokenizing, set the description for tqdm's progress bar
            self.dataframe['temp_col'] = self.apply_with_progress(self.dataframe['temp_col'], self.tokenize_string,
                                                                  'Tokenizing...')
            # self.dataframe['temp_col'] = self.dataframe['temp_col'].progress_apply(self.tokenize_string)
            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(self.default_remove_punctuations)
            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(self.remove_numbers)
            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(self.to_lowercase)
            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(self.unicode_normalize)
            self.dataframe['temp_col'] = self.dataframe['temp_col'].apply(' '.join)
            self.dataframe = self.remove_duplicates(self.dataframe, 'temp_col', self.deduplication_threshold)

            # Drop the temporary column after deduplication
            self.dataframe.drop(columns = ['temp_col'], inplace = True)
        return self.dataframe

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
            return df.drop_duplicates(subset=[column])
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
            if any(self.is_similar(current_string, checked_string, threshold_percentage) for checked_string in checked_strings):
                continue
            checked_strings.append(current_string)
            unique_rows.append(row)

        return pd.DataFrame(unique_rows)

    @staticmethod
    def tokenize_string(text: str) -> list[str]:
        """Tokenizes the text in DataFrame."""

        return word_tokenize(text)

    def remove_numbers(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if not self.pure_num_pattern.match(token)]

    def custom_remove_punctuations(self, tokens: list[str]) -> list[str]:
        """
        Removes punctuation from the text in DataFrame column/s,
        this is for regular cleaning process.
        :param tokens:
        :return:
        """
        return [self.custom_punctuation_pattern.sub("", token) for token in tokens]

    def default_remove_punctuations(self, tokens: list[str]) -> list[str]:
        """
        Removes punctuation from the text in DataFrame column/s,
        this is for deduplication process.
        :param tokens:
        :return:
        """
        return [self.default_punctuation_pattern.sub("", token) for token in tokens]

    @staticmethod
    def to_lowercase(tokens: list[str]) -> list[str]:
        """Converts all strings to lowercase."""
        return [token.lower() for token in tokens]

    @staticmethod
    def unicode_normalize(tokens: list[str]) -> list[str]:
        """Perform Unicode normalization."""
        return [normalize('NFKD', token) for token in tokens]

    @staticmethod
    def apply_with_progress(df: pd.DataFrame, function, description) -> pd.DataFrame:
        tqdm.pandas(desc = description)
        df = df.progress_apply(function)
        return df
