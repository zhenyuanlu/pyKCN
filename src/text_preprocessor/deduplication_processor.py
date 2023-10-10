import string
import pandas as pd
from rapidfuzz import fuzz

from .base_processor import BaseProcessor


class DeduplicationProcessor(BaseProcessor):
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
        """

        :param dataframe:
        :param columns_to_process:
        :param columns_to_deduplicate:
        :param date_column:
        :param custom_delimiter:
        :param custom_punctuation:
        :param default_punctuation:
        :param deduplication_threshold:
        :param word_len_threshold:
        :param fill_na:
        """
        super().__init__(dataframe,
                         columns_to_process,
                         columns_to_deduplicate,
                         date_column,
                         custom_delimiter,
                         custom_punctuation,
                         default_punctuation,
                         deduplication_threshold,
                         word_len_threshold,
                         fill_na)

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

    def execute_processor(self) -> pd.DataFrame:
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

