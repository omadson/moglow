import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

class SeriesDataset(Dataset):
    """
    A PyTorch Dataset object to represent time series data with conditional subsequences and labels.

    Parameters
    ----------
    data : np.ndarray
        The data containing the time series. It should be of shape (num_samples, num_steps, num_variables).
    labels : np.ndarray
        The labels for each sample in `data`. It should be of shape (num_samples,).
    series : np.ndarray
        An array containing the series index for each sample in `data`. It should be of shape (num_samples,).
    tau : int, optional
        An integer representing the time shift used to create the conditional subsequences. Default is 3.

    Attributes
    ----------
    x : np.ndarray
        The time series data, converted to a PyTorch tensor of shape (num_samples, num_variables, final_sequence_length).
    cond : np.ndarray
        The conditional subsequences for each sample, converted to a PyTorch tensor of shape 
        (num_samples, num_conditional_variables, final_sequence_length).
    labels : np.ndarray
        The labels for each sample.
    series : np.ndarray
        The series index for each sample.
    """
    def __init__(
            self,
            data: np.ndarray,
            labels: np.ndarray,
            series: np.ndarray,
            tau: int = 3
        ):
        num_samples, num_steps, num_variables = data.shape
        self.x = data.swapaxes(1, 2)[:, :, tau:]
        final_sequence_lenght = self.x.shape[2]
        self.cond  = (
            sliding_window_view(data[:, :-1, :], (1, tau, num_variables))
            .squeeze()
            .reshape(num_samples, final_sequence_lenght, -1)
            .swapaxes(2, 1)
        )
        self.labels = labels
        self.series = series
            
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
        """
        Return the sample at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the time series data, the conditional subsequences, the label and the series index 
            for the sample at the given index.
        """
        return {
                'x': self.x[idx,:,:],
                'cond': self.cond[idx,:,:],
                'label': self.labels[idx],
                'serie': self.series[idx]
            }