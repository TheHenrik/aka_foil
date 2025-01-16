import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
# import keras

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


def main():
    # Load data
    data = pl.read_csv('data/small_dataset.csv')
    data = data.select(["kulfan_upper_0", "kulfan_upper_1", "kulfan_upper_2", "kulfan_upper_3", "kulfan_upper_4", "kulfan_upper_5", "kulfan_upper_6", "kulfan_upper_7", "kulfan_lower_0", "kulfan_lower_1", "kulfan_lower_2", "kulfan_lower_3", "kulfan_lower_4", "kulfan_lower_5", "kulfan_lower_6", "kulfan_lower_7", "kulfan_LE_weight", "kulfan_TE_thickness"])
    data = data.to_numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Create DataLoader
    dataset = TensorDataset(data_tensor, data_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Autoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Start Training")
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, _ = batch
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    # Save the trained model
    torch.save(model.state_dict(), 'autoencoder.pth')


def test():
    pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('weights/autoencoder.pth', map_location=device))
    model.eval()

    # Load test data
    test_data = pl.read_csv('data/cleaned_data.csv')
    test_data = test_data.select(["kulfan_upper_0", "kulfan_upper_1", "kulfan_upper_2", "kulfan_upper_3", "kulfan_upper_4", "kulfan_upper_5", "kulfan_upper_6", "kulfan_upper_7", "kulfan_lower_0", "kulfan_lower_1", "kulfan_lower_2", "kulfan_lower_3", "kulfan_lower_4", "kulfan_lower_5", "kulfan_lower_6", "kulfan_lower_7", "kulfan_LE_weight", "kulfan_TE_thickness"])
    test_data = test_data.to_numpy()
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

    # Run the model on the test data
    with torch.no_grad():
        for i in range(10):
            data = test_tensor[i]
            data = data.unsqueeze(0)
            reconstructed = model(data)
            print(reconstructed.cpu().numpy())
            print(data)



# Example usage
if __name__ == "__main__":
    main()