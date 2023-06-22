from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, DataLoader


def split_sequences(data: np.ndarray, labels: np.ndarray, sequence_length: int = 3, overlap: bool=True):
    """Split a dataset of time series into subsequences of a specified length.

    Parameters
    ----------
    data : np.ndarray
        The time series dataset, where the first axis corresponds to the
        time steps, the second axis corresponds to the samples, and the
        third axis corresponds to the variables.
    labels : np.ndarray
        The labels for each sample in the time series dataset.
    sequence_length : int, optional
        The length of each subsequence. Defaults to 3.
    overlap : bool, optional
        Whether to create overlapping subsequences. Defaults to True.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - 'data': an np.ndarray containing the subsequences of the time
        series data.
        - 'labels': an np.ndarray containing the corresponding labels for
        each subsequence.
        - 'series': an np.ndarray containing the index of the series from
        which each subsequence was extracted.

    Notes
    -----
    This function uses a sliding window approach to extract the subsequences
    from the time series data. If `overlap=True`, then the subsequences
    will overlap. If `overlap=False`, then the subsequences will not overlap.

    """
    num_series, num_steps, num_variables = data.shape
    if num_steps < sequence_length:
        raise Exception(f"sequence_length must be into 1 to {num_steps}.")
    if overlap:
        # Calculate the number of variables (i.e., features) in the data
        series = np.arange(num_series)
        subsequences = (
            sliding_window_view(data, (1, sequence_length, num_variables))
            .squeeze()
        )
        num_subsequences_by_sequence = subsequences.shape[1]
        subsequences_labels = labels.repeat(num_subsequences_by_sequence)
        subsequences_series = series.repeat(num_subsequences_by_sequence)
        subsequences = subsequences.reshape((-1, sequence_length, num_variables))#.swapaxes(1,2)
    else:
        # TODO: implement no overlap option
        ...
    return {
        'data': subsequences,
        'labels': subsequences_labels,
        'series': subsequences_series
    }


class ExperimentDataset(Dataset):
    def __init__(
            self,
            data: np.ndarray,
            tau: int = 3,
            name: str = None
        ):
        self.name = name
        num_samples, num_steps, num_variables = data.shape
        self.x = data.swapaxes(1, 2)[:, :, tau:]
        final_sequence_lenght = self.x.shape[2]
        self.cond  = torch.tensor(
            sliding_window_view(data[:, :-1, :], (1, tau, num_variables))
            .squeeze()
            .reshape(num_samples, -1, num_variables)
            # .swapaxes(1, 0)
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': self.x[idx,:,:],
            'cond': self.cond[idx,:,:],
        }
    
    @property
    def info(self):
        num_features, sequence_length = self[0]['x'].shape
        num_conditional_features, _ = self[0]['cond'].shape
        return {
            'num_features': num_features, 
            'sequence_length': sequence_length,
            'num_conditional_features': num_conditional_features,
            'name': self.name
        }

output_folder = Path('data/processed').absolute()

def cut_array(percentage, arr):
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]

def convert_to_windows(data, sequence_length, dimensions=2):
    windows = []; w_size = sequence_length
    for i, g in enumerate(data): 
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if dimensions == 2 else w.view(-1))
    return torch.stack(windows)

def load_dataset(dataset, folder=None, less=False):
    if folder:
        folder = Path(folder).absolute() / dataset
    else:
        folder = Path(output_folder) / dataset
    if not folder.exists():
        raise Exception(f'Processed Data not found. {folder}')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(folder / f'{file}.npy'))
    # loader = [i[:, debug:debug+1] for i in loader]
    if less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def load_data(name: str, sequence_length: int = 10, folder=None, valid_set=True):
    train_loader, test_loader, labels = load_dataset(name, folder)
    train, test = next(iter(train_loader)), next(iter(test_loader))
    original_train, original_test = train, test
    window_train = torch.DoubleTensor(convert_to_windows(
        train,
        sequence_length=sequence_length
    ))
    window_test = torch.DoubleTensor(convert_to_windows(
        test,
        sequence_length=sequence_length
    ))
    window_train_set_original = ExperimentDataset(window_train, tau=sequence_length-1, name=name)
    window_valid_set = None
    if valid_set:
        window_train_set, window_valid_set = (
            torch
            .utils
            .data
            .random_split(window_train_set_original, [.8, .2])
        )
        window_train_set.info = data_info(window_train_set, name=name)
        window_valid_set.info = data_info(window_valid_set, name=name)
    window_test_set = ExperimentDataset(window_test, tau=sequence_length-1, name=name)

    return {
        'original_train': original_train,
        'original_test': original_test,
        'window_train_original': window_train_set_original,
        'window_train': window_train_set,
        'window_test': window_test_set,
        'window_valid': window_valid_set,
        'labels': labels
    }
    
def data_info(dataset, name):
    num_features, sequence_length = dataset[0]['x'].shape
    tao, num_conditional_features = dataset[0]['cond'].shape
    return {
        'num_features': num_features, 
        'sequence_length': sequence_length,
        'num_conditional_features': int(num_conditional_features * tao),
        'name': name
    }