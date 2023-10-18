"""
DeduplicationProcessor
======================
The `DeduplicationProcessor` module offers a systematic approach to reduce duplicates in textual data.
Building upon the foundation of the `BaseProcessor`, this class defines a specific sequence of steps
tailored for data deduplication.

Description:
------------
The `DeduplicationProcessor` is designed to preprocess textual data with the primary goal of making
it suitable for deduplication. By applying a series of normalization and transformation steps, this processor
ensures that minor variations in the textual data (like casing, punctuation, or encoding differences) do not
lead to unnecessary duplicates. The processed data can then be compared for deduplication using string similarity
or other deduplication techniques.

Procedure:
------
**DEDUPLICATION PIPELINE**:

- Remove Angle Bracket Patterns
- Convert Text to Lowercase
- Apply Unicode Normalization
- Remove Non-ASCII Characters
- Remove Punctuation (default set)
- Remove Numbers (all types by default)

After processing through these steps, textual data is standardized to a format where direct string comparisons
become more reliable, thereby aiding in deduplication efforts.

Note:
Tokenization might not be essential for the deduplication purpose in this specific context.

*Example Transformation*::

    # Original text
    'Machine-learning method; lstm; déép learning ; <sub>27</sub>'
    # Remove angle bracket patterns
    -> 'Machine-learning method; lstm; déép learning'
    # Convert Text to Lowercase
    -> 'machine-learning method; lstm; déép learning'
    # Apply Unicode Normalization, e.g. é, french accents to english, unicode '\u00e9'
    -> 'machine-learning method; lstm; deep learning'
    # Remove Non-ASCII Characters, e.g. symbols, emojis, etc.
    -> 'machine-learning method; lstm; deep learning'
    # Remove Punctuation
    -> 'machine learning method lstm deep learning'
    # Remove Numbers
    -> 'machine learning method lstm deep learning'


Usage:
--------
**Example 1: Basic Usage**::

    deduplication_processor = DeduplicationProcessor(dataframe,
                                    columns_to_process = ['column_1', 'column_2'])
    processed_df = deduplication_processor.execute_processor()



**Example 2: Load Data Extraction Pipeline Data and Save Deduplicated Data**::

    from src.utils.utils import load_data_from_prep, save_data_from_prep

    PARENT_PATH = r'path/to/parent/folder'
    CACHE_LOCATION = r'path/to/cache/folder'
    DATA_TYPE = 'extracted'
    PIPELINE_NAME = 'PIPELINE_NAME'

    dataframe = load_data_from_prep(pipeline_name = PIPELINE_NAME,
                                     data_type = DATA_TYPE,
                                     root_path = PARENT_PATH,
                                     filename = None)

    deduplication_processor = DeduplicationProcessor(dataframe,
                                    columns_to_process = ['column_1', 'column_2'],
                                    cache_location = 'CACHE_LOCATION',
                                    cache_format = 'csv')
    processed_df = deduplication_processor.execute_processor()

    save_data_from_prep(processed_df, pipeline_name = PIPELINE_NAME,
                        data_type = 'deduplicated',
                        root_path = PARENT_PATH)


"""
import pandas as pd
from rapidfuzz import fuzz

from .base_processor import BaseProcessor


class DeduplicationProcessor(BaseProcessor):

    DEDUPLICATION_STEPS = [
        {"description": "Removing Angle Bracket Pattern...", "function": "remove_angle_brackets", "args": {}},
        {"description": "Lowering Cases...", "function": "to_lowercase", "args": {}},
        {"description": "Unicode Normalization...", "function": "unicode_normalize", "args": {}},
        {"description": "Removing Non ASCII...", "function": "remove_non_ascii", "args": {}},
        # {"description": "Tokenizing...", "function": "tokenize_string", "args": {}},
        {"description": "Removing Punctuation...", "function": "remove_punctuation",
         "args": {"punctuation_type": "default"}},
        {"description": "Removing Numbers...", "function": "remove_numbers", "args": {"pattern_type": "all"}},
    ]

    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns_to_deduplicate: list[str] = None,
                 deduplication_threshold: float = 1.0,
                 deduplication_steps: list[dict] = None,
                 new_col_name = 'temp_col',
                 fill_na: str = None):
        super().__init__(dataframe = dataframe,
                         columns_to_deduplicate = columns_to_deduplicate,
                         deduplication_threshold = deduplication_threshold,
                         fill_na = fill_na)
        self.handle_nan(mode_type = 'deduplication')
        self.pipeline = deduplication_steps or self.DEDUPLICATION_STEPS
        self.new_col_name = new_col_name

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
        self.combine_columns(self.columns_to_deduplicate, new_col_name = self.new_col_name)
        new_col_name = self.new_col_name

        if new_col_name in self.dataframe.columns:
            for step in self.pipeline:
                description = step["description"]
                function = getattr(self, step["function"])
                args = step["args"]

                self.dataframe[new_col_name] = self.apply_with_progress(self.dataframe[new_col_name],
                                                                        function,
                                                                        description,
                                                                        **args)

            # Join tokens and handle deduplication after processing
            # self.dataframe[new_col_name] = self.dataframe[new_col_name].apply(' '.join)
            self.dataframe = self.remove_duplicates(self.dataframe, new_col_name, self.deduplication_threshold)

            # Drop the temporary column after deduplication
            self.dataframe.drop(columns = [new_col_name], inplace = True)
            # self.dataframe.drop(columns = ['col_1', 'col_2'], inplace = True)
        return self.dataframe
