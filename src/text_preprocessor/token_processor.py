import re
import string
import pandas as pd
from tqdm import tqdm
from unicodedata import normalize
from collections import defaultdict

from base_processor import BaseProcessor


class TokenProcessor(BaseProcessor):
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

    def split_by_delimiter(self, text):
        """Split string by the specified delimiters."""
        return re.split(self.delimiter_pattern, text)


    def update_vocabulary_and_dictionary(self, original_tokens, stemmed_tokens):
        for orig, stem in zip(original_tokens, stemmed_tokens):
            self.vocabulary.add(orig)
            self.stem_to_original[stem].add(orig)

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


from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pandas as pd


class TextProcessor(BaseProcessor):
    def __init__(self, dataframe, columns_to_process, word_len_threshold = 2):
        super().__init__(dataframe, columns_to_process)
        self.word_len_threshold = word_len_threshold
        self.vocabulary = set()
        self.stem_to_original = defaultdict(set)
        self.stemmer = PorterStemmer()

    def split_by_delimiter(self, text):
        return re.split(r'[;,-/]', text)

    def handle_hyphenated_terms(self, tokens):
        return [term for token in tokens for term in re.split(r'-', token)]

    def strip_whitespace(self, tokens):
        return [token.strip() for token in tokens]

    def remove_numericals(self, tokens):
        return [token for token in tokens if not token.isdigit()]

    def filter_by_length(self, tokens):
        return [token for token in tokens if len(token) > self.word_len_threshold]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def update_vocabulary_and_dictionary(self, original_tokens, stemmed_tokens):
        for orig, stem in zip(original_tokens, stemmed_tokens):
            self.vocabulary.add(orig)
            self.stem_to_original[stem].add(orig)

    def process_column(self, column_name):
        # Get the column data
        column_data = self.dataframe[column_name]

        # List to store processed data
        processed_data = []

        # Iterate over each record in the column
        for record in column_data:
            # 1. Split by delimiter
            tokens = self.split_by_delimiter(record)

            # 2. Tokenize content
            tokens = word_tokenize(' '.join(tokens))

            # 3. Handle hyphenated terms
            tokens = self.handle_hyphenated_terms(tokens)

            # 4. Strip whitespace
            tokens = self.strip_whitespace(tokens)

            # 5. Remove numerical values
            tokens = self.remove_numericals(tokens)

            # 6. Filter by length
            tokens = self.filter_by_length(tokens)

            # 7. Convert to lowercase
            tokens = self.to_lowercase(tokens)

            # 8. Remove punctuations
            tokens = self.remove_punctuation(tokens)

            # Store original tokens for updating vocabulary and dictionary
            original_tokens = tokens.copy()

            # 9. Stem tokens
            stemmed_tokens = self.stem_tokens(tokens)

            # 10. Final cleanup
            original_tokens = self.strip_whitespace(original_tokens)
            stemmed_tokens = self.strip_whitespace(stemmed_tokens)

            # 11. Join tokens
            original_record = ', '.join(original_tokens)
            stemmed_record = ', '.join(stemmed_tokens)

            # Update vocabulary and dictionary
            self.update_vocabulary_and_dictionary(original_tokens, stemmed_tokens)

            # Append processed record to processed_data
            processed_data.append(original_record)

        # Update the dataframe column with processed data
        self.dataframe[column_name] = processed_data

        return self.dataframe

# Note: I've not included the implementation of the parent class (BaseProcessor) here,
# as it's expected to be in the provided 'base_processor.py'.
