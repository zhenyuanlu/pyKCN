"""
NCBIFetcher Module
==================
This module provides the NCBIFetcher class for interacting with the NCBI databases, downloading records,
and saving the data to a CSV format. It utilizes the Entrez Programming Utilities (E-utilities) API for
fetching the data from NCBI databases such as PubMed.

The class is designed to handle the querying and downloading of data in batches, which is particularly useful
for large datasets. It also provides functionality to convert the downloaded data into a more accessible CSV format,
using customizable column names based on the data of interest.

Classes:
--------
- NCBIFetcher: A class to interact with NCBI databases, download records, and save data to CSV.

Example:
--------
# Example usage of NCBIFetcher to download data from PubMed and PMC and save it as CSV.

from ncbi_fetcher import NCBIFetcher

# Set up the necessary parameters
email = "your@email.com"
query = "(example[TI] OR example[OT] OR example[AB])"
databases = ['database1', 'database2']
root_path = r"to\your\desired\path\ncbi_data"

# Initialize the NCBIFetcher with the specified parameters
fetcher = NCBIFetcher(email=email,
                      query=query,
                      database=databases,
                      root_path=root_path)

# Query database counts (optional)
# counts = fetcher.query_database_counts()
# print(counts)

# Download the data from the databases
fetcher.download_pubmed_data()

# Save the downloaded data to CSV
fetcher.save_data_to_csv(column_names=DEFAULT_COLUMNS)
"""

# TODO add timestamp to the filename
# TODO make the code more modular

import os
import pandas as pd
from Bio import Entrez, Medline


class NCBIFetcher:
    """
    A class to interact with NCBI databases, download records, and save data to CSV.

    Attributes:
        email (str): The email address used for NCBI interaction.
        query (str): The query string for fetching data.
        database (list[str]| str): List of databases or a single database to query.
        root_path (str): The root path for saving downloaded data.
        output_txt_path (str): The path to save raw text data from NCBI.
        output_csv_path (str): The path to save processed data in CSV format.

    Raises:
        ValueError: If 'email' or 'database' is not provided.
    """

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
        # Set the email for NCBI interactions
        Entrez.email = email

    def query_database_counts(self) -> dict:
        """
        Query the NCBI databases to count the number of articles matching the query.
        :return: A dictionary with database names and their respective article count.
        """
        handle = Entrez.egquery(term=self.query)
        record = Entrez.read(handle)
        handle.close()
        return {row['DbName']: row['Count'] for row in record['eGQueryResult']}

    def download_pubmed_data(self, batch_size: int = 10000) -> None:
        """
        Download PubMed data in batches.
        :param batch_size: The number of articles to download in each batch.
        :return: None
        """
        for db in self.database:
            output_path = os.path.join(self.root_path, f"{db}_{self.output_txt_path}")
            handle = Entrez.esearch(db=db, term=self.query, usehistory='y', idtype='acc')
            record = Entrez.read(handle)
            handle.close()
            webenv = record['WebEnv']
            query_key = record['QueryKey']
            count = int(record['Count'])

            with open(output_path, 'w') as out_handle:
                for start in range(0, count, batch_size):
                    end = min(count, start + batch_size)
                    fetch_handle = Entrez.efetch(db=db, rettype='medline', retmode='text',
                                                 retstart=start, retmax=batch_size, webenv=webenv,
                                                 query_key=query_key, idtype='acc')
                    data = fetch_handle.read()
                    fetch_handle.close()
                    out_handle.write(data)

    def save_data_to_csv(self, column_names: list[str] = None) -> None:
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
