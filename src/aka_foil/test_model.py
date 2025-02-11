import torch
from aka_foil.train_simple_model import MLP
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from aka_foil.airfoil_handling_functions import get_kulfan_parameters, get_file_coordinates, get_data_from_xfoil
from aka_foil.split_data import data_input_to_model_input, model_output_to_data_output


# Define analysis
airfoil = 'ag36'
airfoil_path = os.path.join(Path(__file__).parent.parent.parent, "data", "airfoils", f"{airfoil}.dat")
alphas = np.linspace(-30, 30, num=100)
re = 4e5
mach = 0
n_crit = 8
xtr_upper = 1
xtr_lower = 1

# Transform Airfoil and Operating Point into input format
kulfan_parameters = get_kulfan_parameters(get_file_coordinates(airfoil_path))
lower_weights = torch.tensor(kulfan_parameters['lower_weights'], dtype=torch.float32)
upper_weights = torch.tensor(kulfan_parameters['upper_weights'], dtype=torch.float32)
te_thickness = torch.tensor([kulfan_parameters['TE_thickness']], dtype=torch.float32)
leading_edge_weight = torch.tensor([kulfan_parameters['leading_edge_weight']], dtype=torch.float32)
Re = torch.tensor([re], dtype=torch.float32)
mach = torch.tensor([mach], dtype=torch.float32)
n_crit = torch.tensor([n_crit], dtype=torch.float32)
xtr_upper = torch.tensor([xtr_upper], dtype=torch.float32)
xtr_lower = torch.tensor([xtr_lower], dtype=torch.float32)

inputs = []
for alpha in alphas:
    alpha_tensor = torch.tensor([alpha], dtype=torch.float32)
    input = torch.cat([lower_weights, upper_weights, leading_edge_weight, te_thickness,
                   alpha_tensor, Re, mach, n_crit, xtr_upper, xtr_lower])
    inputs.append(input)
input = torch.stack(inputs)
input = data_input_to_model_input(input)

# Load the model from best checkpoint
model_state_dict = torch.load('../../weights/checkpoint_41200.pth', map_location=torch.device('cpu'))
net = MLP()
net.load_state_dict(model_state_dict['model_state_dict'])
net.eval()

# Get the results from the neural network
results = model_output_to_data_output(net(input))
# Get the results from xfoil surrogate (for comparison)
xfoil_results = get_data_from_xfoil(airfoil, re, num_points=20)

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# Drag
axes[0, 0].plot(results[:, 2], results[:, 1], label='aka_foil', color='blue')
axes[0, 0].scatter(xfoil_results[:, 2], xfoil_results[:, 1], label='xfoil', marker='x')
axes[0, 0].set_title("Drag Polar")
axes[0, 0].set_xlabel(r"$C_D$")
axes[0, 0].set_ylabel(r"$C_L$")
axes[0, 0].set_xlim(0, 0.05)
axes[0, 0].set_ylim(-0.5, 1.7)
axes[0, 0].grid()
axes[0, 0].legend()
# Lift
axes[0, 1].plot(alphas, results[:, 1], label='aka_foil', color='orange')
axes[0, 1].scatter(xfoil_results[:, 0], xfoil_results[:, 1], label='xfoil', marker='x')
axes[0, 1].set_title("Lift Polar")
axes[0, 1].set_xlabel(r"$\alpha$")
axes[0, 1].set_ylabel(r"$C_L$")
axes[0, 1].set_xlim(-5, 15)
axes[0, 1].set_ylim(-0.5, 1.7)
axes[0, 1].grid()
axes[0, 1].legend()
# Moment
axes[1, 0].plot(results[:, 1], results[:, 3], label='aka_foil', color='orange')
axes[1, 0].scatter(xfoil_results[:, 1], xfoil_results[:, 3], label='xfoil', marker='x')
axes[1, 0].set_title("Moment Polar")
axes[1, 0].set_xlabel(r"$C_L$")
axes[1, 0].set_ylabel(r"$C_M$")
axes[1, 0].set_xlim(-0.5, 1.7)
axes[1, 0].set_ylim(-0.1, 0.)
axes[1, 0].grid()
axes[1, 0].legend()
# Transition
axes[1, 1].plot(results[:, 4], results[:, 1], label='aka_foil', color='orange')
axes[1, 1].scatter(xfoil_results[:, 4], xfoil_results[:, 1], label='xfoil', marker='x', color='orange')
axes[1, 1].plot(results[:, 5], results[:, 1], label='aka_foil', color='blue')
axes[1, 1].scatter(xfoil_results[:, 5], xfoil_results[:, 1], label='xfoil', marker='x', color='blue')
axes[1, 1].set_title("Transition Location Polar")
axes[1, 1].set_xlabel(r"$x$")
axes[1, 1].set_ylabel(r"$C_L$")
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(-0.5, 1.7)
axes[1, 1].grid()
axes[1, 1].legend()
# Layout anpassen und anzeigen
plt.tight_layout()
plt.show()

