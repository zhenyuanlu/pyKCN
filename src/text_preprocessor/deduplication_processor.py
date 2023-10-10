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

from base_processor import BaseProcessor


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


