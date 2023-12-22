"""
DEFAULT_QUERY = "(cancer[TI] OR cancer[OT] OR cancer[AB])"
DEFAULT_BATCH_SIZE = 10000
DEFAULT_COLUMNS = ['TI', 'OT', 'AB', 'DP']
"""
import os
import pandas as pd
from Bio import Entrez, Medline


class NCBIFetcher:

    def __init__(self,
                 email: str = None,
                 query: str = None,
                 database: list[str] | str = None,
                 root_path: str = None,
                 output_txt_path: str = 'output.txt',
                 output_csv_path: str = 'output.csv'):
        if not email:
            raise ValueError('Email is required for NCBI interactions.')
        if not database:
            raise ValueError('At least one database must be specified.')
        self.email = email
        self.query = query
        self.database = database
        self.output_txt_path = output_txt_path
        self.output_csv_path = output_csv_path
        self.root_path = root_path
        Entrez.email = email  # Set the email for NCBI

    def query_database_counts(self) -> dict:
        """
        Count the number of articles in each database.
        :return: A dictionary with database names and their respective article count.
        """
        handle = Entrez.egquery(term=self.query)
        record = Entrez.read(handle)
        handle.close()
        return {row['DbName']: row['Count'] for row in record['eGQueryResult']}

    def download_pubmed_data(self, batch_size: int = 10000):
        """
        Download PubMed data in batches.
        :param batch_size: The number of articles to download in each batch.
        :return: None
        """
        for db in self.database:
            output_path = os.path.join(self.root_path, f"{db}_{self.output_txt_path}")
            handle = Entrez.esearch(db = db, term = self.query, usehistory = 'y', idtype = 'acc')
            record = Entrez.read(handle)
            handle.close()
            webenv = record['WebEnv']
            query_key = record['QueryKey']
            count = int(record['Count'])

            with open(output_path, 'w') as out_handle:
                for start in range(0, count, batch_size):
                    end = min(count, start + batch_size)
                    fetch_handle = Entrez.efetch(db = db, rettype = 'medline', retmode = 'text',
                                                 retstart = start, retmax = batch_size, webenv = webenv,
                                                 query_key = query_key, idtype = 'acc')
                    data = fetch_handle.read()
                    fetch_handle.close()
                    out_handle.write(data)

    def save_data_to_csv(self, column_names: list[str] = None):
        if column_names is None:
            column_names = ['TI', 'OT', 'AB', 'DP']

        for db in self.database:
            input_dir = self.root_path
            output_file = os.path.join(self.root_path, f"{db}_{self.output_csv_path}")
            prefix = f"{db}_output"
            article = []

            for filename in os.listdir(input_dir):
                if filename.endswith('.txt') and filename.startswith(prefix):
                    file_path = os.path.join(input_dir, filename)
                    with open(file_path, encoding='utf8') as handle:
                        records = Medline.parse(handle)
                        for record in records:
                            article.append({k: record.get(k, '') for k in column_names})

            df = pd.DataFrame(article)
            df.to_csv(output_file, encoding='utf-8', index=False)

    # TODO add timestamp to the filename

