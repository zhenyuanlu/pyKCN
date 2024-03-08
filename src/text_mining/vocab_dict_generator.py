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
import re


class VocabDictBuilder:
    def __init__(self, dataframe,
                 original_col: str = 'original_data',
                 stemmed_col: str = 'stemmed_data') -> None:
        """
        Initialize the VocabDictBuilder with the given DataFrame.

        :param dataframe: DataFrame containing the original and stemmed keywords.
        :param original_col: the name of the column with original keywords.
        :param stemmed_col: the name of the column with stemmed keywords.
        """
        self.dataframe = dataframe
        self.original_col = original_col
        self.stemmed_col = stemmed_col
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
            original_keywords = self._ensure_list_format(row[self.original_col], is_original=True)
            stemmed_keywords = self._ensure_list_format(row[self.stemmed_col])
            for stemmed_keyword, original_keyword in zip(stemmed_keywords, original_keywords):
                self.stemmed_to_originals[stemmed_keyword].add(original_keyword)
                self.all_stemmed[stemmed_keyword] += 1

    @staticmethod
    def _ensure_list_format(keywords, is_original=False) -> list:
        """
        Ensure the keywords are in list format. If 'is_original' is True, handle various separators for original keywords.

        :param keywords: Keywords data, could be a string or a list.
        :param is_original: Flag to indicate if the keywords are original keywords.
        :return: A list of keywords.
        """
        if isinstance(keywords, str) and is_original:
            # If it's a string of original keywords, attempt to split by common separators
            return re.split(r'[;,\s]\s*', keywords)
        elif isinstance(keywords, (list, np.ndarray)):
            return list(keywords)
        elif isinstance(keywords, str):
            # Assuming a single keyword in a string
            return [keywords]
        else:
            raise ValueError(f"Unsupported data structure: {type(keywords)}")

    def _create_vocabulary(self) -> set:
        """
        Create a set of unique stemmed keywords based on gathered data.

        :return: Set of stemmed keywords.
        """
        return set(self.all_stemmed.keys())

    def _create_dictionary(self) -> dict:
        """
        Create a mapping from stemmed keywords to original keywords based on gathered data.

        :return: Dictionary mapping stemmed to sets of original keywords.
        """
        return dict(self.stemmed_to_originals)