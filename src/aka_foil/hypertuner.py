from ray import tune
from ray.tune.schedulers import ASHAScheduler

import torch.nn as nn
import torch.optim as optim

from aka_foil.split_data import create_train_data, data_loader
from pathlib import Path
import torch
import ray

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, n_hidden_layers):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation_fn())
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the training function
def train_mlp(config, data):
    train_loader, test_loader, val_loader = data_loader(data=data, batch_size=32)
    input_size = data[0].shape[1]
    output_size = data[1].shape[1]

    model = MLP(input_size=input_size, 
                hidden_size=config['hidden_sizes'], 
                output_size=output_size, 
                activation_fn=config['activation_fn'],
                n_hidden_layers=config['n_hidden_layers'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)

    # Evaluate on validation data
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)



def hyper_tune():
    # Define the hyperparameter search space
    search_space = {
        "hidden_sizes": tune.choice([16, 32, 64, 128]),
        "n_hidden_layers": tune.choice([3, 4, 5]),
        "activation_fn": tune.choice([nn.SELU, nn.SiLU]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": tune.choice([20, 40, 60])
    }

    # Define the scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2)
    
    data = create_train_data(Path('data/small_dataset.csv'))

    # Run the hyperparameter search
    tune.run(
        tune.with_parameters(train_mlp, data=data),
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=search_space,
        num_samples=10,
        scheduler=scheduler)


if __name__ == "__main__":
    hyper_tune()
