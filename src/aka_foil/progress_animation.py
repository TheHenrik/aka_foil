import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from aka_foil.train_simple_model import MLP
from aka_foil.airfoil_handling_functions import get_kulfan_parameters, get_file_coordinates, get_data_from_xfoil
from aka_foil.split_data import data_input_to_model_input, model_output_to_data_output
import os
from pathlib import Path


# Define analysis
airfoil = 'ag25'
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


# Get the results from xfoil surrogate (for comparison)
xfoil_results = get_data_from_xfoil(airfoil, re, num_points=20)
# Create a figure and axis
fig, ax = plt.subplots()

# Set up the plot limits
ax.set_xlim(0, 0.05)
ax.set_ylim(-0.5, 1.5)
line, = ax.plot([], [], label='aka_foil', color='blue')
ax.scatter(xfoil_results[:, 2], xfoil_results[:, 1], label='xfoil', marker='x')
ax.set_title("Drag Polar")
ax.set_xlabel(r"$C_D$")
ax.set_ylabel(r"$C_L$")
ax.grid()
ax.legend()

# Create a line object which will be updated in the animation
# line, = ax.plot([], [], lw=2)

# Initialization function to set up the background of each frame
def init():
    line.set_data([], [])
    return line,

# Animation function which updates the figure
def animate(i):
    file = f'checkpoints_2025-02-07/checkpoint_{i}.pth'
    model_state_dict = torch.load(file, map_location=torch.device('cpu'))
    net = MLP()
    net.load_state_dict(model_state_dict['model_state_dict'])
    net.eval()

    # Get the results from the neural network
    results = model_output_to_data_output(net(input))
    line.set_data(results[:, 2], results[:, 1])
    return line, 

def animate_ray(i):
    file = f'/var/home/tjalf/ray_results/train_model_2025-02-09_23-30-37/train_model_7cbfa_00196_196_activation_fn=ref_ph_615fc99a,hidden_size=256,lr=0.0005,n_hidden_layers=5_2025-02-09_23-30-38/checkpoint_{i:06d}/checkpoint.pt'
    model_state_dict = torch.load(file, map_location=torch.device('cpu'))
    net = MLP()
    net.load_state_dict(model_state_dict[0])
    net.eval()

# Create the animation object
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=range(11700, 15000, 100), interval=500, blit=True)

# Display the animation
plt.show()
# Save the animation
ani.save('/var/home/tjalf/Projects/aka_foil/progress_animation.mp4', writer='ffmpeg')