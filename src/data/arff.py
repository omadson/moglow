from pathlib import Path
from typing import List, Tuple, Optional

import yaml

import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy.io import arff

from src.data import SeriesDataset
from src.data.utils import split_sequences


def read_arff(
        data_name: str, 
        serie_column: str,
        label_column: str, 
        label_tuple: List[str],
        train: bool = True, 
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a dataset in ARFF format and convert it to NumPy arrays.

    Parameters
    ----------
    data_name : str
        The name of the dataset to load (without the '.arff' extension).
    serie_column : str
        The name of the column containing the series data in the ARFF file.
    label_column : str
        The name of the column containing the labels in the ARFF file.
    label_tuple : List[str]
        A list of unique label names in the order they appear in the ARFF file.
    train : bool, optional
        A boolean flag indicating whether to read the training data or the test data.
        Default is True (read the training data).
        

    Returns
    -------
    dict
        A dictionary containing two keys: 'data' and 'labels'.
        The 'data' value is a 3D NumPy array with shape (num_samples, sequence_length, num_variables),
        where `num_samples` is the number of series in the ARFF file, `sequence_length` is the
        length of each series, and `num_variables` is the number of variables in each series.
        The 'labels' value is a 1D NumPy array with shape (num_samples,) containing the label index
        for each series in the 'data' array, as determined by the `label_tuple`.
    """ 
    # Specify the path to the ARFF file,
    # load the data from the ARFF file, and
    # convert the data to a pandas Dataframe
    data_path = (
        Path('../data/raw/') /
        data_name /
        ('train.arff' if train else 'test.arff')
    )
    array, _ = arff.loadarff(data_path)
    df = pd.DataFrame(array)
    # Iterate through each row of the DataFrame to create
    # tensors for series and a vector for the labels
    series = []
    labels = []
    for _, row in df.iterrows():
        sample = np.stack([np.array(row.tolist()) for row in row[serie_column]])
        series.append(sample)
        labels.append(label_tuple.index(row[label_column].decode('utf-8')))
    data = np.stack(series).swapaxes(1,2)
    labels = np.stack(labels)
    return {
        'data': data,
        'labels': labels
    }


class ArffData:
    def __init__(
            self,
            dataset_name: str,
            train: bool = True,
            sequence_length: int = 10,
            tau: int = 3, 
            contamination: Optional[float] = None,
            random_state: int = 42
        ):
        """
        A class for reading and preprocessing data from .arff files. 

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset file to read.
        train : bool, optional
            Whether to read the training or test data. Default is True.
        sequence_length : int, optional
            The length of subsequences to extract from the data. Default is 10.
        tau : int, optional
            The time lag between subsequences. Default is 3.
        contamination : float or None, optional
            The percentage of anomalous samples to add to the data. Default is None (no contamination).
        random_state : int, optional
            The random seed for reproducibility. Default is 42.

        Properties:
        ----------
        original_data : dict
            A dictionary containing the original data read from the .arff file. The dictionary has two
            keys, 'data' and 'labels', which contain the features and labels of the data, respectively. 
            If contamination is applied, this property will contain a subset of the original data with 
            the specified percentage of anomalous samples.
        subsequences : dict
            A dictionary containing subsequences generated from the original data. The dictionary has two keys,
            'data' and 'labels', which contain the subsequences and their corresponding labels, respectively. 
            The subsequences are generated by sliding a window of length `sequence_length` over the original 
            data with a time lag of `tau`.
        set : SeriesDataset
            A `SeriesDataset` object containing the subsequences generated from the original data. This object
            is used for training and testing anomaly detection models. It contains the subsequences, their
            corresponding labels, and the time lags between them (specified by `tau`).

        Methods:
        --------
        __init__(self, dataset_name, train=True, sequence_length=10, tau=3, contamination=None, random_state=42)
            Initializes the ArffData object by reading the configuration file and data, and generating subsequences.

        """
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
        self.original_data = read_arff(**data_configuration, train=train)
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