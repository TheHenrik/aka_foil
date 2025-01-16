from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_train_data(file_path: Path, test_size: float = 0.1, val_size: float = 0.1, seed: int = None):
    # Load the data
    df = pl.read_csv(file_path)
    
    # Shuffle the data
    df: pl.DataFrame = df.sample(fraction=1, seed=seed)

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

    