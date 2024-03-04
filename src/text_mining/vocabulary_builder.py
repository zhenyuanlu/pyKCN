"""
# Example usage:
# dataframe = pd.DataFrame(...)
# vocab_builder = VocabularyBuilder(dataframe)
# vocabulary = vocab_builder.vocabulary
# dictionary = vocab_builder.dictionary
"""


import pandas as pd
from collections import defaultdict
import ast

class VocabularyBuilder:
    def __init__(self, dataframe: pd.DataFrame, vocab_column: str = 'stemmed_data', dict_columns: list = None):
        """
        Initialize the VocabularyBuilder with the given DataFrame and columns.

        :param dataframe: DataFrame containing the text data.
        :param vocab_column: The name of the column containing the stemmed keywords for the vocabulary.
        :param dict_columns: List of two column names, [stemmed column, original column],
                             for building the dictionary.
        """
        if dict_columns is None:
            dict_columns = [vocab_column, 'original_data']

        # Error handling for the input parameters
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The provided dataframe is not a valid pandas DataFrame.")
        if not vocab_column in dataframe.columns:
            raise ValueError(f"The vocab_column '{vocab_column}' is not in the DataFrame.")
        if not all(col in dataframe.columns for col in dict_columns):
            raise ValueError(f"One or more dict_columns '{dict_columns}' are not in the DataFrame.")

        self.dataframe = dataframe
        self.vocab_column = vocab_column
        self.dict_columns = dict_columns

        self.vocabulary = self.build_vocabulary()
        self.dictionary = self.build_dictionary()

    def build_vocabulary(self):
        """
        Build the vocabulary from the stemmed keywords.

        :return: Set of unique stemmed keywords.
        """
        vocab = set()
        try:
            for keywords_list_str in self.dataframe[self.vocab_column].dropna():
                # Convert the string representation of the list to an actual list
                # Using `ast.literal_eval` is safe for literal structures
                keywords_list = ast.literal_eval(keywords_list_str)
                # Ensure that it's a list before updating the vocabulary
                if isinstance(keywords_list, list):
                    vocab.update(keywords_list)
                else:
                    raise ValueError(f"Row in column '{self.vocab_column}' is not a list.")
        except ValueError as e:
            # This will catch errors from `ast.literal_eval`
            raise ValueError(f"Error parsing string to list in column '{self.vocab_column}': {e}")
        except Exception as e:
            raise ValueError(f"Error building vocabulary from column '{self.vocab_column}': {e}")
        return vocab

    def build_dictionary(self):
        """
        Build the dictionary mapping stemmed keywords to a list of original keywords.

        :return: Dictionary where keys are stemmed keywords and values are lists of original keywords.
        """
        dictionary = defaultdict(set)
        stemmed_col, original_col = self.dict_columns
        try:
            for _, row in self.dataframe[[stemmed_col, original_col]].dropna().iterrows():
                stemmed_keywords = row[stemmed_col]
                original_keywords = row[original_col]
                for stemmed_keyword in stemmed_keywords:
                    dictionary[stemmed_keyword].add(original_keywords)
        except Exception as e:
            raise ValueError(f"Error building dictionary from columns '{self.dict_columns}': {e}")
        dictionary = {key: list(value) for key, value in dictionary.items()}
        return dictionary


