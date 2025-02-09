from airfoil_handling_functions import *
from split_data import *
import torch
from train_simple_model import MLP
import torch.nn as nn

# Paths
airfoil_path = '../../data/airfoils/ag40.dat'
# model_save_path = 'bigtry/checkpoint_17325.pth'

# Model config
hidden_sizes = 512
n_hidden_layers = 5
activation_function = nn.SiLU
input_size = 25
output_size = 6

# Operating point
alphas = np.linspace(-30, 30, num=100)
Re = 2e5
mach = 0
n_crit = 9
xtr_upper = 1
xtr_lower = 1

# Transform Airfoil and Operating Point into input format
kulfan_parameters = get_kulfan_parameters(get_file_coordinates(airfoil_path))

print(kulfan_parameters)

lower_weights = torch.tensor(kulfan_parameters['lower_weights'], dtype=torch.float32)
upper_weights = torch.tensor(kulfan_parameters['upper_weights'], dtype=torch.float32)
te_thickness = torch.tensor([kulfan_parameters['TE_thickness']], dtype=torch.float32)
leading_edge_weight = torch.tensor([kulfan_parameters['leading_edge_weight']], dtype=torch.float32)
# alpha = torch.tensor([alpha], dtype=torch.float32)
Re = torch.tensor([Re], dtype=torch.float32)
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

# Reload model and evaluate for input
# Lade das gespeicherte Tupel
# model_state_dict, optimizer_state_dict = torch.load(model_save_path)
model_state_dict = torch.load('checkpoint_41200.pth')
#epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict = torch.load('bigtry/checkpoint_17325.pth')
net = MLP()

# Lade die Parameter in das Modell
net.load_state_dict(model_state_dict['model_state_dict'])

# Setze das Modell in den Evaluierungsmodus
net.eval()

y = net(input)
y = model_output_to_data_output(y)
print(y)

results = y
print(results.shape)

analysis_confidence = results[:, 0]
CL = results[:, 1]
CD = results[:, 2]
CM = results[:, 3]
Top_Xtr = results[:, 4]
Bottom_Xtr = results[:, 5]

# Get XFOIL Results from .csv
# Pfad zur CSV-Datei
file_path = "../../data/airfoils/ag40_T1_Re0.200_M0.00_N9.0.csv"

# Datei einlesen
ref_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
ref_alpha = ref_data[:, 0]
ref_CL = ref_data[:, 1]
ref_CD = ref_data[:, 2]
ref_CM = ref_data[:, 4]
ref_Top_Xtr = ref_data[:, 5]
ref_Bottom_Xtr = ref_data[:, 6]
print(ref_data.shape)

import matplotlib.pyplot as plt

# Subplots erstellen (1 Zeile, 2 Spalten)
fig, axes = plt.subplots(2, 2, figsize=(12, 5))

# Drag
axes[0, 0].plot(CD, CL, label='aka_foil', color='blue')
axes[0, 0].scatter(ref_CD, ref_CL, label='xfoil', marker='x')
axes[0, 0].set_title("Drag Polar")
axes[0, 0].set_xlabel(r"$C_D$")
axes[0, 0].set_ylabel(r"$C_L$")
axes[0, 0].grid()
axes[0, 0].legend()

# Lift
axes[0, 1].plot(alphas, CL, label='aka_foil', color='orange')
axes[0, 1].scatter(ref_alpha, ref_CL, label='xfoil', marker='x')
axes[0, 1].set_title("Lift Polar")
axes[0, 1].set_xlabel(r"$\alpha$")
axes[0, 1].set_ylabel(r"$C_L$")
axes[0, 1].grid()
axes[0, 1].legend()

# Moment
axes[1, 0].plot(CL, CM, label='aka_foil', color='orange')
axes[1, 0].scatter(ref_CL, ref_CM, label='xfoil', marker='x')
axes[1, 0].set_title("Moment Polar")
axes[1, 0].set_xlabel(r"$C_L$")
axes[1, 0].set_ylabel(r"$C_M$")
axes[1, 0].grid()
axes[1, 0].legend()

# Transition
axes[1, 1].plot(Top_Xtr, CL, label='aka_foil', color='orange')
axes[1, 1].scatter(ref_Top_Xtr, ref_CL, label='xfoil', marker='x', color='orange')
axes[1, 1].plot(Bottom_Xtr, CL, label='aka_foil', color='blue')
axes[1, 1].scatter(ref_Bottom_Xtr, ref_CL, label='xfoil', marker='x', color='blue')
axes[1, 1].set_title("Transition Location Polar")
axes[1, 1].set_xlabel(r"$C_L$")
axes[1, 1].set_ylabel(r"$C_M$")
axes[1, 1].grid()
axes[1, 1].legend()

# Layout anpassen und anzeigen
plt.tight_layout()
plt.show()

