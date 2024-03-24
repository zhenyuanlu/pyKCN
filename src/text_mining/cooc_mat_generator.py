import pandas as pd
from scipy.sparse import dok_matrix, csr_matrix
from itertools import combinations
from tqdm import tqdm


class CoocMatrixGenerator:
    def __init__(self, vocabulary: dict, threshold: int = 1):
        self.vocab = vocabulary
        self.threshold = threshold
        self.vocab_size = len(vocabulary)
        self.matrix = None

    def generate_matrix(self, documents: pd.Series):
        dok_mat = dok_matrix((self.vocab_size, self.vocab_size), dtype = int)

        for document in tqdm(documents, desc = "Building co-occurrence matrix"):
            indices = [self.vocab.get(word) for word in document if word in self.vocab]
            for i, j in combinations(set(filter(None, indices)), 2):
                dok_mat[i, j] += 1
                dok_mat[j, i] += 1

        self.matrix = dok_mat.tocsr()

        if self.threshold > 1:
            self.matrix.data[self.matrix.data < self.threshold] = 0
            self.matrix.eliminate_zeros()

        print(f"CSR Matrix dimensions: {self.matrix.shape}")
        return self.vocab, self.matrix

    def print_matrix_snippet(self, num_rows: int = 10, num_cols: int = 10):
        if not self.matrix:
            print("Matrix has not been generated yet.")
            return

        # Fetching the subset of the matrix and vocabulary
        row_indices, col_indices = self.matrix.nonzero()
        snippet = self.matrix[row_indices[:num_rows], col_indices[:num_cols]]

        # Get the corresponding words from the vocabulary
        vocab_list = list(self.vocab.keys())
        row_words = [vocab_list[i] for i in row_indices[:num_rows]]
        col_words = [vocab_list[i] for i in col_indices[:num_cols]]

        # Create a DataFrame for display
        df_snippet = pd.DataFrame(snippet.toarray(), index = row_words, columns = col_words)
        print(df_snippet)
