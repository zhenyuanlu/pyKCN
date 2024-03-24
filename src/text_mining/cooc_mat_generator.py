from scipy.sparse import dok_matrix, csr_matrix
from itertools import combinations


class CoocMatrixGenerator:
    def __init__(self, vocabulary: dict, threshold: int = 1):
        """
        Initialize the co-occurrence matrix generator.

        :param vocabulary: A dictionary mapping keywords to their respective indices in the matrix.
        :param threshold: The minimum co-occurrence count to include in the matrix.
        """
        self.vocab = vocabulary
        self.threshold = threshold
        self.vocab_size = len(vocabulary)

    def generate_matrix(self, documents) -> csr_matrix:
        """
        Generate a co-occurrence matrix from the given documents.

        :param documents: A pandas Series where each row contains a list of keywords.
        :return: A Compressed Sparse Row (CSR) co-occurrence matrix.
        """
        # Use dok_matrix for easy incremental updates
        dok_mat = dok_matrix((self.vocab_size, self.vocab_size), dtype=int)

        for document in documents:
            indices = [self.vocab.get(word) for word in document if word in self.vocab]
            for i, j in combinations(set(filter(None, indices)), 2):
                dok_mat[i, j] += 1
                dok_mat[j, i] += 1  # Ensure the matrix is symmetric

        # Convert to CSR format for efficient arithmetic operations and slicing
        csr_mat = dok_mat.tocsr()

        # Filter out co-occurrences below the threshold
        if self.threshold > 1:
            csr_mat.data[csr_mat.data < self.threshold] = 0
            csr_mat.eliminate_zeros()

        return csr_mat
