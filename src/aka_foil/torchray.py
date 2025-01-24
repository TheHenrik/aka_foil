import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
# from filelock import FileLock
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from aka_foil.split_data import create_train_data, data_loader
from pathlib import Path
import ray

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_fn, n_hidden_layers):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(activation_fn())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

def train_aka_foil(config, data):
    train_loader, test_loader, val_loader = data_loader(data=data, batch_size=2**13)
    input_size = data[0].shape[1]
    output_size = data[1].shape[1]


    net = MLP(input_size=input_size, 
                hidden_size=config['hidden_sizes'], 
                output_size=output_size, 
                activation_fn=config['activation_fn'],
                n_hidden_layers=config['n_hidden_layers'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    for epoch in range(1_000):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_func(outputs, labels, device=device)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
            #                                     running_loss / epoch_steps))
            #     running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)

                loss = loss_func(outputs, labels, device=device)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps)},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def test_best_model(best_result, data):    
    _, test_loader, _ = data_loader(data=data, batch_size=32)
    input_size = data[0].shape[1]
    output_size = data[1].shape[1]
    
    
    best_trained_model = MLP(input_size=input_size, 
                hidden_size=best_result.config['hidden_sizes'], 
                output_size=output_size, 
                activation_fn=best_result.config['activation_fn'],
                n_hidden_layers=best_result.config['n_hidden_layers'])
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))


def loss_func(pred: torch.Tensor, actual: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
    weight = torch.tensor([0, 1, 1, 1, 1, 1]).to(device)
    
    diff = pred - actual
    diff = diff.nan_to_num_(0)
    diff = diff * weight
    return torch.mean(diff ** 2)


def main(num_samples=10, max_num_epochs=10, smoke_test=False):
    data = create_train_data(Path("../../data/small_dataset.csv"))
    
    config = {
        "hidden_sizes": tune.choice([64]),
        "n_hidden_layers": tune.choice([5]),
        "activation_fn": tune.choice([nn.SELU]),
        "lr": tune.loguniform(0.007, 0.007),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=100,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_aka_foil, data=data),
            resources={"cpu": 10, "gpu": 0}
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
    main(num_samples=1, max_num_epochs=500)
