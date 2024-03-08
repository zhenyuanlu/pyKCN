"""
# TODO - Add type hints and docstrings
# Example usage
import pandas as pd
# df = pd.DataFrame([...])
# keyword_processor = VocabDictBuilder(df)
# vocabulary = keyword_processor.vocabulary  # Access the vocabulary
# dictionary = keyword_processor.dictionary  # Access the dictionary
"""
from collections import defaultdict, Counter
import numpy as np


class VocabDictBuilder:
    def __init__(self, dataframe,
                 original_col: str = 'original_data',
                 stemmed_col: str = 'stemmed_data') -> None:
        """
        Initialize the KeywordProcessor with the given DataFrame.

        :param dataframe: pandas DataFrame containing the original and stemmed keywords.
        :param original_col: the name of the column with original keywords.
        :param stemmed_col: the name of the column with stemmed keywords.
        """
        self.dataframe = dataframe
        self.original_col = original_col
        self.stemmed_col = stemmed_col
        # Preprocessing to gather data for both vocabulary and dictionary in one pass
        self.stemmed_to_originals = defaultdict(set)
        self.all_stemmed = Counter()
        self._gather_data()
        self.vocabulary = self._create_vocabulary()
        self.dictionary = self._create_dictionary()

    def _gather_data(self) -> None:
        """
        Gather data for both vocabulary and dictionary from the DataFrame in one pass.
        """
        for _, row in self.dataframe.iterrows():
            original_keywords = self._split_keywords(row[self.original_col])
            stemmed_keywords = self._split_keywords(row[self.stemmed_col])
            for stemmed_keyword, original_keyword in zip(stemmed_keywords, original_keywords):
                self.stemmed_to_originals[stemmed_keyword].add(original_keyword)
                self.all_stemmed[stemmed_keyword] += 1

    @staticmethod
    def _split_keywords(keywords) -> list:
        """
        Helper function to split keywords based on the data structure.

        :param keywords: numpy array or list of keywords.
        :return: list of keywords.
        """
        if isinstance(keywords, list):
            return keywords
        elif isinstance(keywords, np.ndarray):
            return keywords.tolist()
        else:
            raise ValueError(f"Unsupported data structure: {type(keywords)}")

    def _create_vocabulary(self) -> set:
        """
        Create a set of unique stemmed keywords based on gathered data.

        :return: set of stemmed keywords.
        """
        return set(self.all_stemmed.keys())

    def _create_dictionary(self) -> dict:
        """
        Create a mapping from stemmed keywords to original keywords based on gathered data.

        :return: dictionary mapping stemmed to original keywords.
        """
        # Convert sets to lists for consistency
        return {stemmed: {original} for stemmed, originals in self.stemmed_to_originals.items() for original in
                originals}
