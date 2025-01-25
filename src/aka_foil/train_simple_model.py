import torch.nn as nn
import torch
from split_data import *

input_size = 24
output_size = 6
hidden_size = 48
n_hidden_layers = 1
activation_fn = nn.SiLU
learning_rate = 1e-3
criterion = nn.functional.mse_loss

data = create_train_data(Path('../../data/small_dataset.csv'), test_size=0.2, val_size=0.)
train_loader, test_loader, val_loader = data_loader(data=data, batch_size=2048)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation_fn())
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=0.2)) # TODO Dropout layer
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    num_epochs = 1_000
    for epoch in range(num_epochs):
        # Training
        model.train() # Sets model in training mode
        train_loss = 0.
        for x, y in train_loader:
            x, y, = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

    # Testphase
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), 'current_model.pth')

if __name__ == '__main__':
    train_model()
