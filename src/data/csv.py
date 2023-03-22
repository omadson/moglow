from pathlib import Path
from typing import List, Tuple, Optional, Dict

import yaml

import pandas as pd
import numpy as np
from numpy.random import default_rng

from src.data import SeriesDataset
from src.data.utils import split_sequences


def read_csv(
        data_name: str, 
        serie_column: str,
        label_column: str,
        data_columns: List[str],
        label_tuple: List[str],
        train: bool = True
    ) -> Dict:
    """Reads a CSV file and returns a dictionary containing the data and labels.
    
    Args:
    - data_name (str): Name of the CSV file.
    - serie_column (str): Name of the column containing the series identifier.
    - label_column (str): Name of the column containing the label for each series.
    - data_columns (List[str]): List of column names containing the data for each series.
    - label_tuple (List[str]): List of possible label values.
    - train (bool): Indicates whether to read the training or test data. Defaults to True.
    
    Returns:
    - A dictionary with two keys:
        - 'data': A 3D numpy array of shape (n_samples, n_frames, n_features), where:
            - n_samples: Number of unique series in the CSV file.
            - n_frames: Number of rows per series.
            - n_features: Number of columns in the data_columns list.
        - 'labels': A 1D numpy array of shape (n_samples,), where each element represents the label for a series.
    """
    data_path = (
        Path('../data/raw/') /
        data_name /
        ('train.csv' if train else 'test.csv')
    )
    df = pd.read_csv(data_path)


    aggregated_series = (
        df
        .groupby([serie_column, label_column])
        .agg(list)
        .reset_index()
    )
    serie_example_label = aggregated_series[serie_column][0]
    labels = (
        aggregated_series[label_column]
        .apply(lambda x: label_tuple.index(x))
        .values
    )
    n_samples = aggregated_series.shape[0]
    n_frames = df.query(f'{serie_column} == @serie_example_label').shape[0]
    n_features = len(data_columns)
    data = df[data_columns].values.reshape(n_samples, n_frames, n_features)
    return {
        'data': data,
        'labels': labels
    }



class CSVData:
    def __init__(
            self,
            dataset_name: str,
            train: bool = True,
            sequence_length: int = 10,
            tau: int = 3, 
            contamination: Optional[float] = None,
            random_state: int = 42
        ):
        self.dataset_name = dataset_name
        self.sequence_length = sequence_length
        self.tau = tau
        self.contamination = contamination
        self.random_state = random_state
        # read configurations
        config_file_path = (
            Path(f'../experiments/configuration/datasets/') / 
            dataset_name
        ).with_suffix('.yaml')
        with open(config_file_path, 'r') as f:    # Load 
            data_configuration = yaml.safe_load(f)
        # read data
        data_configuration
        self.original_data = read_csv(**data_configuration, train=train)
        if self.contamination:
            rng = default_rng(seed=random_state)
            nominal_samples = np.where(self.original_data['labels'] == 0)[0]
            anomaly_samples = np.where(self.original_data['labels'] == 1)[0]
            max_contamination = len(anomaly_samples) / (len(anomaly_samples) + len(nominal_samples))
            # make sure that the specified contamination is not greater than the proportion of anomalies in the data
            if self.contamination > max_contamination:
                raise Exception(
                    f"Contamination must be into [0, {max_contamination:.3f}] "
                    "for this dataset."
                )
            num_anomaly = int(anomaly_samples.shape[0]*self.contamination/(1-self.contamination))
            # randomly select a subset of anomalies to add to the data
            final_anomaly_samples = rng.choice(anomaly_samples, num_anomaly)
            # concatenate the indices of nominal samples and the selected anomalies
            samples = np.concatenate([nominal_samples, final_anomaly_samples], axis=0)
            # select the corresponding data and labels using the selected indices
            self.original_data = {
                'data': self.original_data['data'][samples],
                'labels': self.original_data['labels'][samples]
            }
        # split the data into subsequences
        self.subsequences = split_sequences(**self.original_data, sequence_length=sequence_length)
        # create a SeriesDataset object from the subsequences
        self.set = SeriesDataset(**self.subsequences, tau=tau)
    
    def info(self):
        num_features, sequence_length = self.set[0]['x'].shape
        num_conditional_features, _ = self.set[0]['cond'].shape
        return {
            'num_features': num_features, 
            'sequence_length': sequence_length,
            'num_conditional_features': num_conditional_features
        }