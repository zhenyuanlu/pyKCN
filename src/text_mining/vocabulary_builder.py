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
import numpy as np


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
        for keywords_array in self.dataframe[self.vocab_column].dropna():
            # Check if the data is in an array-like format and convert to list if necessary
            if isinstance(keywords_array, list):
                keywords_list = keywords_array
            elif hasattr(keywords_array, 'tolist'):  # Check for NumPy arrays or similar
                keywords_list = keywords_array.tolist()
            else:
                raise TypeError(f"Unsupported data type in column '{self.vocab_column}': {type(keywords_array)}")

            for phrase in keywords_list:
                # Assuming each phrase is a string of keywords separated by spaces
                words = phrase.split()
                vocab.update(words)
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
                # Ensure stemmed_keywords is a list of strings (individual keywords)
                stemmed_keywords = np.atleast_1d(row[stemmed_col]).tolist()
                # Ensure original_keywords is processed as a list
                original_keywords = np.atleast_1d(row[original_col]).tolist()

                for stemmed_keyword in stemmed_keywords:
                    # Here we ensure stemmed_keyword is treated as a string for hashing purposes
                    if isinstance(stemmed_keyword, list):
                        # This should not happen if your data is structured correctly;
                        # stemmed_keywords should be a single keyword string per list element.
                        continue  # or handle appropriately

                    # Adding each original keyword to the set for this stemmed keyword
                    for original_keyword in original_keywords:
                        dictionary[stemmed_keyword].add(original_keyword)
        except Exception as e:
            raise ValueError(f"Error building dictionary from columns '{self.dict_columns}': {e}")

        # Convert sets to lists for consistent output
        dictionary = {key: list(value) for key, value in dictionary.items()}
        return dictionary




