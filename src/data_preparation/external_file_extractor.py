"""

================

"""


import os
import pandas as pd
from .base_extractor import BaseExtractor


class ExternalFileExtractor(BaseExtractor):
    """
    Subclass for extracting data from CSV/Excel files.
    """
    def __init__(self,
                 data_mapping,
                 data_dir,
                 new_column_names = None,
                 date_type = 'year'):
        super().__init__()
        self.data_mapping = data_mapping
        self.new_column_names = new_column_names
        self.data_dir = data_dir
        self.date_type = 'year'

    def load_data(self):
        all_data = []
        final_corpus_columns = None
        final_date_column = None
        for folder, config in self.data_mapping.items():
            corpus_columns = config.get('corpus_columns')
            date_column = config.get('date_column')
            folder_path = os.path.join(self.data_dir, folder)
            target_columns = corpus_columns + [date_column]

            # Default column names if new_column_names is None
            if not self.new_column_names:
                new_column_names = {col: f"col_{i + 1}" for i, col in enumerate(corpus_columns)}
                new_column_names[date_column] = 'date_col'
            else:
                new_column_names = self.new_column_names


            # Getting new corpus column names
            final_corpus_columns = [new_column_names[col] for col in corpus_columns]
            # Getting the new date column name
            final_date_column = new_column_names[date_column]

            data_frames = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                df = None
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path, usecols = target_columns, on_bad_lines='skip')
                    elif file.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(file_path, usecols = target_columns, header=0)
                except Exception as e:
                    print(f"An error occurred while reading file {file_path}: {e}")

                if df is not None:
                    # Rename columns
                    df = df[target_columns]
                    df.rename(columns = new_column_names, inplace = True)
                    data_frames.append(df)
            all_data_frames = pd.concat(data_frames, ignore_index = True)
            all_data.append(all_data_frames)
        final_df = pd.concat(all_data, ignore_index = True)
        # Get the final column names from the DataFrame
        # all_columns = final_df.columns.tolist()
        # final_corpus_columns = all_columns[:-1]  # All columns except the last
        # final_date_column = all_columns[-1]  # The last column
        return final_df, final_corpus_columns, final_date_column




