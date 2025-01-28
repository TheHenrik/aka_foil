import torch.nn as nn
import torch
from split_data import *
from datetime import datetime
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 256 * 2**3
data = create_train_data(Path('../../data/small_dataset_v3.csv'), test_size=0.2, val_size=0., use_physical_transformations=True)
train_loader, test_loader, val_loader = data_loader(data=data, batch_size=batch_size)

input_size = data[0].shape[1]
output_size = data[1].shape[1]
hidden_size = 64
n_hidden_layers = 3
activation_fn = nn.SiLU
learning_rate = 1e-4
#criterion = nn.functional.mse_loss

# Prepare the loss function
loss_weights = torch.ones(output_size, dtype=torch.float32).to(device)
loss_weights[0] *= 0.  # Analysis confidence
loss_weights[1] *= 3  # CL
loss_weights[2] *= 5  # ln(CD)
loss_weights[3] *= 0.2  # CM
loss_weights[4] *= 0.1  # Top Xtr
loss_weights[5] *= 0.1  # Bot Xtr

loss_weights = loss_weights / torch.sum(loss_weights) * 1000

def criterion(y_hat, y):
    unweighted_loss = torch.mean(
            torch.nn.functional.mse_loss(
                y_hat, y,
                reduction='none',
            ),
            dim=0
        )
    return torch.sum(unweighted_loss * loss_weights)



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
            #layers.append(nn.Dropout(p=0.2)) # TODO Dropout layer
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Normal forward pass
        y = self.network(x)

        # Physics informed approach:
        # Mirrored Airfoil around x axis (with mirrored AoA) has the same (now negative) lift coefficient
        # and the same drag coefficient, same with moment coefficient and transition locations
        x_flipped = x.clone()
        x_flipped[:, :8] = -1 * x[:, 8:16] # flip kulfan of upper and lower side
        x_flipped[:, 8:16] = -1 * x[:, :8] # flip kulfan of upper and lower side
        x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
        x_flipped[:, 18] = -1 * x[:, 18]  # flip alpha TODO
        x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower TODO
        x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper TODO

        # Evaluate network again for flipped condition
        y_flipped = self.network(x_flipped)

        # Re-flip the results
        y_unflipped = y_flipped.clone()
        y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
        y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
        y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
        y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

        # Fuse the results (arithmetic mean)
        y_fused = (y + y_unflipped) / 2

        y_fused[:, 4] = torch.clip(y_fused[:, 4].clone(), 0, 1)  # Top_Xtr clipped to range
        y_fused[:, 5] = torch.clip(y_fused[:, 5].clone(), 0, 1)  # Bot_Xtr clipped to range

        return y_fused

def train_model():
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Model
    model = MLP()
    model.to(device)
    print(model)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=50,
        verbose=True,
    )

    # Reload Checkpoint
    checkpoint_folder_path = None
    if (checkpoint_folder_path != None) and (os.path.exists(checkpoint_folder_path)):
        all_files = os.listdir(checkpoint_folder_path)
        checkpoint_files = [f for f in all_files if f.startswith("checkpoint") and f.endswith(".pth")]
        epochs = []
        for file in checkpoint_files:
            try:
                epoch = int(file.split("_")[1].split(".")[0])
                epochs.append(epoch)
            except (IndexError, ValueError):
                pass
        start_epoch = max(epochs)
    else:
        start_epoch = 0
        today = datetime.today().strftime('%Y-%m-%d')
        checkpoint_folder_path = f"checkpoints_{today}"
        os.makedirs(checkpoint_folder_path, exist_ok=True)

    num_epochs = 1_000_000
    save_every_x_epochs = 5
    for epoch in range(start_epoch, num_epochs):
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
        print(f"Epoch {epoch},  Train Loss: {train_loss:.4f}")

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
        print(f"          Test Loss: {test_loss:.4f}")

        scheduler.step(test_loss)

        if epoch % save_every_x_epochs == 0:
            checkpoint_path = checkpoint_folder_path + f"/checkpoint_{str(epoch)}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    train_model()
