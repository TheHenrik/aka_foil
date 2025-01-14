import torch
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18, 16),
            nn.SELU(),
            nn.Linear(16, 14),
            nn.SELU(),
            nn.Linear(14, 12),
            nn.SELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 14),
            nn.SELU(),
            nn.Linear(14, 16),
            nn.SELU(),
            nn.Linear(16, 18),
            nn.SELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage
if __name__ == "__main__":
    # Load data
    data = pl.read_csv('data/cleaned_data.csv')
    data = data.select(pl.col(pl.Int64).first(18)).to_numpy()

    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Start Training")
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, _ = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'autoencoder.pth')
