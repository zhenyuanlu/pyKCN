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
from urllib.error import HTTPError, URLError
import time


class NCBIFetcher:
    """
    A class to interact with NCBI databases, download records, and save data to CSV.

    Attributes:
        email (str): The email address used for NCBI interaction.
        api_key (str): The API key used for NCBI interaction.
        query (str): The query string for fetching data.
        database (list[str] | str): List of databases or a single database to query.
        root_path (str): The root path for saving downloaded data.
        output_txt_path (str): The path to save raw text data from NCBI.
        output_csv_path (str): The path to save processed data in CSV format.
        start_year (int): The start year for the search query.
        end_year (int): The end year for the search query.

    Raises:
        ValueError: If 'email', 'api_key', or 'database' is not provided.
    """

    def __init__(self,
                 email: str = None,
                 api_key: str = None,
                 query: str = None,
                 database: list[str] | str = None,
                 root_path: str = None,
                 output_txt_path: str = 'output.txt',
                 output_csv_path: str = 'output.csv',
                 start_year: int = None,
                 end_year: int = None):

        if not email:
            raise ValueError('Email is required for NCBI interactions.')
        if not api_key:
            raise ValueError('API key is required for NCBI interactions.')
        if not database:
            raise ValueError('At least one database must be specified.')
        self.email = email
        self.api_key = api_key
        self.query = query
        self.database = database if isinstance(database, list) else [database]
        self.output_txt_path = output_txt_path
        self.output_csv_path = output_csv_path
        self.root_path = root_path
        self.start_year = start_year
        self.end_year = end_year
        Entrez.email = email
        Entrez.api_key = api_key

    def query_database_counts(self) -> dict:
        """
        Query the specified NCBI databases to get the count of records for the given query.
        :return: A dictionary containing the database names and the count of records.
        """
        handle = Entrez.egquery(term=self.query)
        record = Entrez.read(handle)
        handle.close()
        return {row['DbName']: row['Count'] for row in record['eGQueryResult']}

    def download_pubmed_data(self, batch_size: int = 1000, date_range: int = 1) -> None:
        """
        Download the data from the specified NCBI databases in batches.
        :param batch_size: The batch size for fetching records. Default is 1000.
        :param date_range: The range of years to fetch data in a single batch. Default is 1.
        :return: None
        """
        if self.start_year is None or self.end_year is None:
            raise ValueError("Both start_year and end_year must be specified.")

        for db in self.database:
            current_start_year = self.start_year
            while current_start_year <= self.end_year:
                current_end_year = min(current_start_year + date_range - 1, self.end_year)
                query = f"{self.query} AND ({current_start_year}:{current_end_year}[pdat])"
                try:
                    handle = Entrez.esearch(db=db, term=query, usehistory='y', retmax=batch_size)
                    record = Entrez.read(handle)
                    handle.close()
                    webenv = record['WebEnv']
                    query_key = record['QueryKey']
                    count = int(record['Count'])

                    if count == 0:
                        print(f"No records found for database: {db} with query: {query}")
                        current_start_year += date_range
                        continue

                    for start in range(0, count, batch_size):
                        end = min(count, start + batch_size)
                        output_path = os.path.join(self.root_path,
                                                   f"{db}_{current_start_year}-{current_end_year}_batch_{start + 1}_to_{end}.txt")
                        print(
                            f"Fetching records {start + 1} to {end} from {db} for years {current_start_year}-{current_end_year}...")
                        print(f"Using WebEnv: {webenv}, QueryKey: {query_key}, Start: {start}, End: {end}")
                        success = False
                        attempts = 0
                        while not success and attempts < 5:  # Retry logic
                            try:
                                fetch_handle = Entrez.efetch(db=db, rettype='medline', retmode='text',
                                                             retstart=start, retmax=batch_size, webenv=webenv,
                                                             query_key=query_key)
                                data = fetch_handle.read()
                                fetch_handle.close()
                                with open(output_path, 'w', encoding='utf-8') as out_handle:
                                    out_handle.write(data)
                                success = True
                            except HTTPError as e:
                                print(f"HTTPError: {e.code} - {e.reason}. Retrying in 5 seconds...")
                                attempts += 1
                                time.sleep(5)  # Wait before retrying
                            except URLError as e:
                                print(f"URLError: {e.reason}. Retrying in 5 seconds...")
                                attempts += 1
                                time.sleep(5)  # Wait before retrying
                            except Exception as e:
                                print(f"An error occurred: {e}")
                                break
                    print(f"Data successfully written to {self.root_path}")
                except HTTPError as e:
                    print(f"HTTPError: {e.code} - {e.reason}")
                except URLError as e:
                    print(f"URLError: {e.reason}")
                except Exception as e:
                    print(f"An error occurred: {e}")

                current_start_year += date_range

    def save_data_to_csv(self, column_names: list[str] = None) -> None:
        """
        Save the downloaded data to a CSV file with the specified column names.
        :param column_names: The list of column names to include in the CSV file. Default is ['TI', 'OT', 'AB', 'DP'].
        :return: None
        """
        if column_names is None:
            column_names = ['TI', 'OT', 'AB', 'DP']

        for db in self.database:
            input_dir = self.root_path
            output_file = os.path.join(self.root_path, f"{db}_{self.output_csv_path}")
            prefix = f"{db}_"
            article = []

            for filename in os.listdir(input_dir):
                if filename.endswith('.txt') and filename.startswith(prefix):
                    file_path = os.path.join(input_dir, filename)
                    with open(file_path, encoding='utf-8') as handle:
                        records = Medline.parse(handle)
                        for record in records:
                            article.append({k: record.get(k, '') for k in column_names})

            df = pd.DataFrame(article)
            df.to_csv(output_file, encoding='utf-8', index=False)

