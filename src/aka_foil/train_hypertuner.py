""" 
Hypertuning script for the model. Reuses some functions from train_simple_model.py, but is different in some ways to enable better parallelization and hypertuning.
"""


import os
import tempfile
import torch
import torch.nn as nn
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from aka_foil.split_data import create_train_data, data_loader
from pathlib import Path
import ray

from aka_foil.train_simple_model import MLP, criterion


def train_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 2**14
    data = create_train_data(Path('/var/home/tjalf/Projects/aka_foil/data/small_dataset_v3.csv'), test_size=0.2, val_size=0., use_physical_transformations=True)
    train_loader, test_loader, val_loader = data_loader(data=data, batch_size=batch_size)

    input_size = data[0].shape[1]
    output_size = data[1].shape[1]
    hidden_size = config["hidden_size"]
    n_hidden_layers = config["n_hidden_layers"]
    activation_fn =  config["activation_fn"]
    learning_rate = config["lr"]
    #criterion = nn.functional.mse_loss

    # Model
    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_hidden_layers=n_hidden_layers, activation_fn=activation_fn)
    model.to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=50,
        verbose=True,
    )

    for epoch in range(1_000_000):
        # Training
        model.train() # Sets model in training mode
        train_loss = 0.
        for x, y in train_loader:
            x, y, = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y, output_size, device)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Testphase
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y, output_size, device)
                test_loss += loss.item()
        test_loss /= len(test_loader)

        scheduler.step(test_loss)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": test_loss},
                checkpoint=checkpoint,
            )


def main(num_samples=1_000, max_time=1000, smoke_test=False):
    # data = create_train_data(Path("data/small_dataset_v3.csv"))
    
    config = {
        "hidden_size": tune.choice([64, 128, 256, 512,]),
        "n_hidden_layers": tune.choice([3, 4, 5, 6]),
        "activation_fn": tune.choice([nn.SELU, nn.SiLU, nn.ReLU]),
        "lr": tune.loguniform(1e-3, 5e-5),
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_time,
        grace_period=200,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model,),
            resources={"cpu": 10, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    context = ray.init()
    print(context.dashboard_url)
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    # test_best_model(best_result, data=data)


if __name__ == "__main__":
    main(num_samples=200, max_time=1000)
