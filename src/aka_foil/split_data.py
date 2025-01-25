from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_train_data(file_path: Path, test_size: float = 0.1, val_size: float = 0.1, seed: int = None):
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


def data_loader(batch_size: int = 32, data: tuple = None):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

    