import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import string
from base_processor import BaseProcessor


class TokenProcessor(BaseProcessor):
    def __init__(self, dataframe, columns_to_process, word_len_threshold=2, stemming=True, delimiter_pattern=r"[;,-/]"):
        super().__init__(dataframe)
        self.columns_to_process = columns_to_process
        self.word_len_threshold = word_len_threshold
        self.stemming = stemming
        self.delimiter_pattern = delimiter_pattern
        self.stemmer = PorterStemmer()

    def split_by_delimiter(self, text):
        """Split string by the specified delimiters."""
        return re.split(self.delimiter_pattern, text)

    def tokenize_content(self, text):
        """Tokenize using NLTK."""
        return word_tokenize(text)

    def handle_hyphenated_terms(self, tokens):
        """Process hyphenated words."""
        return [word for token in tokens for word in re.split(r'-', token)]

    def strip_whitespace(self, tokens):
        """Strip leading/trailing whitespace from each token."""
        return [token.strip() for token in tokens]

    def remove_numericals(self, tokens):
        """Remove standalone numerical values from the tokens."""
        return [token for token in tokens if not token.isdigit()]

    def filter_by_length(self, tokens):
        """Remove tokens based on length."""
        return [token for token in tokens if len(token) > self.word_len_threshold]

    def convert_to_lowercase(self, tokens):
        """Convert tokens to lowercase."""
        return [token.lower() for token in tokens]

    def remove_punctuations(self, tokens):
        """Remove specified punctuations from tokens."""
        return [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

    def stem_or_lemmatize(self, tokens):
        """Apply stemming."""
        if self.stemming:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens

    def final_cleanup(self, tokens):
        """Final cleanup operations: stripping and removing None values."""
        return [token.strip() for token in tokens if token and token.strip()]

    def join_tokens(self, tokens):
        """Join cleaned tokens back into a single string."""
        return ' '.join(tokens)

    def process_column(self, column_name):
        """Apply all the above methods in the right sequence on a column."""
        col_data = self.dataframe[column_name]
        processed_data = col_data.apply(self.split_by_delimiter) \
                                  .apply(self.tokenize_content) \
                                  .apply(self.handle_hyphenated_terms) \
                                  .apply(self.strip_whitespace) \
                                  .apply(self.remove_numericals) \
                                  .apply(self.filter_by_length) \
                                  .apply(self.convert_to_lowercase) \
                                  .apply(self.remove_punctuations) \
                                  .apply(self.stem_or_lemmatize) \
                                  .apply(self.final_cleanup) \
                                  .apply(self.join_tokens)

        self.dataframe[column_name] = processed_data
        return self.dataframe


