import os
import re
import pandas as pd
import numpy as np
import logging
import json
import glob
import importlib
from datetime import datetime

OUTPUT_DATA_FOLDER_NAME = 'output_data'


def create_output_data_dir(root_path: str) -> str:
    """
    Create an 'output_data' directory in the parent of the 'src' directory.

    :return: Path object representing the 'output_data' directory.
    """
    if root_path is None:
        # Get the current file's directory
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # Find the parent directory
        root_dir = current_dir.parent
    else:
        root_dir = root_path
    # Path to the new 'output_data' directory
    output_data_dir = os.path.join(root_dir, OUTPUT_DATA_FOLDER_NAME)
    # Create the directory if it doesn't exist
    os.makedirs(output_data_dir, exist_ok=True)

    return output_data_dir


def save_data_from_prep(data: pd.DataFrame,
                        pipeline_name: str,
                        data_type: str,
                        root_path: str) -> None:
    """
    Save the extracted data to a file in the 'output_data' directory.

    :param data: The extracted data.
    :param pipeline_name: Name of the data pipeline.
    :param data_type: Type of data (e.g., 'extracted_data', 'removed_duplicates').
    :param root_path: The target path for storing data.
    :return: None
    """
    try:
        # TODO - Create a helper function to handle this
        for col in data.columns:
            if data[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                # Convert column elements directly to lists if they are numpy arrays
                data[col] = data[col].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

        output_data_dir = os.path.join(create_output_data_dir(root_path), pipeline_name)
        os.makedirs(output_data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{timestamp}"
        file_path = os.path.join(output_data_dir, f"{filename}.csv")
        data.to_csv(file_path, index=False)

        metadata = {
            'pipeline_name': pipeline_name,
            'data_type': data_type,
            'filename': f"{data_type}_{timestamp}.csv",
            'path': str(file_path),
            'timestamp': timestamp
        }
        update_metadata_from_prep(metadata, filename, output_data_dir)

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred while saving extracted data: {e}")


def save_data_to_json(data, pipeline_name: str, data_type: str, root_path: str) -> None:
    """
    Save the vocabulary or dictionary data to a JSON file, converting sets to lists for serialization.

    :param data: The data to be saved (vocabulary or dictionary).
    :param pipeline_name: Name of the data pipeline.
    :param data_type: Type of data ('vocab' or 'dict').
    :param root_path: The target path for storing data.
    :return: None
    """
    if not isinstance(data, (set, dict)):
        raise TypeError(f"Data must be a set or dict, not {type(data).__name__}.")

    # Ensure data is in a serializable format (convert sets to lists)
    if isinstance(data, set):
        data = list(data)  # Convert set to list if it's a vocabulary
    elif isinstance(data, dict):
        data = {k: list(v) if isinstance(v, set) else v for k, v in
                data.items()}  # Convert sets to lists in dictionaries

    try:
        output_data_dir = os.path.join(root_path, pipeline_name)
        os.makedirs(output_data_dir, exist_ok = True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cache_{data_type}_{timestamp}.json"
        file_path = os.path.join(output_data_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(data, f, ensure_ascii = False, indent = 4)

        metadata = {
            'pipeline_name': pipeline_name,
            'data_type': data_type,
            'filename': f"{data_type}_{timestamp}.json",
            'path': str(file_path),
            'timestamp': timestamp
        }
        update_metadata_from_prep(metadata, filename, output_data_dir)

        print(f"{data_type.capitalize()} data successfully saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred while saving {data_type} data: {e}")


def update_metadata_from_prep(metadata: dict, filename: str, output_data_dir: str) -> None:
    """
    Update metadata file of the new generated output data.

    :param metadata: Metadata to be added.
    :param output_data_dir: Output data directory for dumping Json file.
    :param filename: Name of the file to name the metadata file after.
    :return: None
    """
    metadata_file = os.path.join(output_data_dir, f'{filename}_metadata.json')
    try:
        with open(metadata_file, 'w') as file:
            json.dump(metadata, file)
    except Exception as e:
        logging.error(f"An error occurred while updating metadata: {e}")


def load_data_from_prep(pipeline_name: str = None,
                        data_type: str = None,
                        root_path: str = None,
                        filename: str = None) -> pd.DataFrame | None:
    """
    Load the extracted data from a file in the 'output_data' directory.

    :param pipeline_name: Name of the data pipeline.
    :param data_type: Type of data (e.g., 'extracted_data', 'removed_duplicates').
    :param root_path: The root path for loading data.
    :param filename: The name of the file to be loaded with extension.
    :return: The loaded data as a pandas DataFrame.
    """
    try:
        if filename:
            file_path = os.path.join(create_output_data_dir(root_path), filename)
        else:
            output_data_dir = os.path.join(create_output_data_dir(root_path), pipeline_name)
            files = glob.glob(os.path.join(output_data_dir, f"{data_type}_*.csv"))

            if not files:
                logging.error(f"No files found with data type '{data_type}' in '{output_data_dir}'")
                return None

            # Extract timestamps from filenames and sort files based on the timestamps
            files.sort(key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(), reverse=True)
            file_path = files[0]  # the first file is the latest file

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            logging.error(f"No such file: '{file_path}'")
            return None

    except Exception as e:
        logging.error(f"An error occurred while loading extracted data: {e}")
        return None


def load_parquet_from_prep(pipeline_name: str = None,
                           data_type: str = None,
                           root_path: str = None,
                           filename: str = None) -> pd.DataFrame | None:
    """
    Load the extracted data from a Parquet file in the 'output_data' directory.

    :param pipeline_name: Name of the data pipeline.
    :param data_type: Type of data (e.g., 'extracted_data', 'removed_duplicates').
    :param root_path: The root path for loading data.
    :param filename: The name of the file to be loaded with extension.
    :return: The loaded data as a pandas DataFrame.
    """
    try:

        if filename:
            file_path = os.path.join(root_path, filename)
        else:
            pipeline_dir = os.path.join(root_path, pipeline_name)
            file_pattern = f"*{data_type}*.parquet"
            files = glob.glob(os.path.join(pipeline_dir, file_pattern))

            if not files:
                logging.error(f"No Parquet files found with data type '{data_type}' in '{pipeline_dir}'")
                return None

            # Extract timestamps from filenames and get the latest file
            timestamp_pattern = r'(\d{8}_\d{6})'
            latest_file = max(files, key=lambda f: re.search(timestamp_pattern, f).group())
            file_path = latest_file

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            return df
        else:
            logging.error(f"No such file: '{file_path}'")
            return None

    except Exception as e:
        logging.error(f"An error occurred while loading Parquet data: {e}")
        return None


def is_package_installed(pkg_name: str, error_msg: str = None) -> bool:
    try:
        importlib.import_module(pkg_name)
        return True
    except ImportError:
        if error_msg:
            raise ImportError(error_msg)
        return False
