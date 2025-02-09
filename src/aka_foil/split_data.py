from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy import pi as _pi

_deg2rad = 180.0 / _pi
_rad2deg = _pi / 180.0

def data_input_to_model_input(inp):
    alpha = inp[:, 18] * _rad2deg

    inp[:, 17] = inp[:, 17] * 50
    inp[:, 18] = np.sin(2 * alpha)
    inp = np.insert(inp, 19, np.cos(alpha), axis=1)
    inp = np.insert(inp, 20, 1 - np.cos(alpha) ** 2, axis=1)
    inp = np.delete(inp, 22, axis=1)
    inp[:, 21] = (np.log(inp[:, 21]) - 12.5) / 3.5
    inp[:, 22] = (inp[:, 22] - 9) / 4.5
    return inp

def data_output_to_model_output(res):
    res[:, 1] = 2 * res[:, 1]
    res[:, 2] = np.log(res[:, 2]) / 2 + 2
    res[:, 3] = 20 * res[:, 3]
    return res

def model_output_to_data_output(res):
    res = res.detach().numpy()
    res[:, 1] = res[:, 1] / 2
    res[:, 2] = np.exp((res[:, 2] - 2) * 2)
    res[:, 3] = res[:, 3] / 20
    return res

def create_train_data(file_path: Path, test_size: float = 0.1, val_size: float = 0.1, seed: int = None, use_physical_transformations=False):
    '''
    Split the data into train, test, and validation sets
    :param file_path: Path of the .csv file
    :param test_size: 0 ... 1 fraction of the data to be used for testing
    :param val_size: 0 ... 1 fraction of the data to be used for validation
    :param seed: Seed for random generator in sampling Default: None
    :return: Tuple of Input data and result data for train, test and validation
            (train_inp, train_res, test_inp, test_res, val_inp, val_res)
    '''
    # Load the data
    df = pl.read_csv(file_path)

    # Remove bad data
    df = df.filter(pl.col("analysis_confidence") != 0)

    # Shuffle the data
    df: pl.DataFrame = df.sample(fraction=1, seed=seed, shuffle=True)

    # Select the first 24 columns
    inp = df.select(df.columns[:24]).to_numpy()
    res = df.select(df.columns[24:]).to_numpy()

    if use_physical_transformations:
        inp = data_input_to_model_input(inp)
        res = data_output_to_model_output(res)

    # Split the data into train, test, and validation sets
    test_size = int(len(inp) * test_size)
    val_size = int(len(inp) * val_size)
    train_size = len(inp) - test_size - val_size

    train_inp = inp[:train_size]
    train_res = res[:train_size]

    test_inp = inp[train_size:train_size + test_size]
    test_res = res[train_size:train_size + test_size]

    val_inp = inp[train_size + test_size:]
    val_res = res[train_size + test_size:]

    return train_inp, train_res, test_inp, test_res, val_inp, val_res


def data_loader(batch_size: int = 32, data: tuple = None, device: torch.device = torch.device('cuda')):
    '''
    Create DataLoader for train, test, and validation sets
    :param batch_size: Size fo the DataLoader batches
    :param data: Tuple of train_inp, train_res, test_inp, test_res, val_inp, val_res
    :return: DataLoader for train, test, and validation sets (train_loader, test_loader, val_loader)
    '''
    train_inp, train_res, test_inp, test_res, val_inp, val_res = data

    # Convert numpy arrays to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(train_inp, dtype=torch.float32), torch.tensor(train_res, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_inp, dtype=torch.float32), torch.tensor(test_res, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_inp, dtype=torch.float32), torch.tensor(val_res, dtype=torch.float32))

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True) 

    return train_loader, test_loader, val_loader
