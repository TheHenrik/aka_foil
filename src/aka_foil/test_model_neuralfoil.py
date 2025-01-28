from airfoil_handling_functions import *
from split_data import *
import torch
from train_simple_model import MLP
import torch.nn as nn

# Paths
airfoil_path = '../../data/airfoils/acc22.dat'
model_save_path = '../../data/train_aka_foil_2025-01-17_13-11-02/' \
                  'train_aka_foil_1f371_00003_3_activation_fn=ref_ph_d09f660c,hidden_sizes=64,lr=0.0070,n_hidden_layers=5_2025-01-17_13-11-03/' \
                  'checkpoint_000499/checkpoint.pt'

# Model config
hidden_sizes = 64
n_hidden_layers = 3
activation_function = nn.SiLU
input_size = 25
output_size = 6

# Operating point
alphas = np.linspace(-30, 30, num=20)
Re = 5e5
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
model_state_dict = torch.load('nn-small.pth')

N_inputs = 25
width = 512
n_hidden_layers = 5
N_outputs = 198
class Net(torch.nn.Module):
    def __init__(self, mean_inputs_scaled, cov_inputs_scaled):
        super().__init__()

        self.mean_inputs_scaled = mean_inputs_scaled
        self.cov_inputs_scaled = cov_inputs_scaled
        self.inv_cov_inputs_scaled = torch.inverse(cov_inputs_scaled)
        self.N_inputs = len(mean_inputs_scaled)

        layers = [
            torch.nn.Linear(N_inputs, width),
            torch.nn.SiLU(),
        ]
        for i in range(n_hidden_layers):
            layers += [
                torch.nn.Linear(width, width),
                torch.nn.SiLU(),
            ]

        layers += [
            torch.nn.Linear(width, N_outputs),
        ]

        self.net = torch.nn.Sequential(*layers)

    def squared_mahalanobis_distance(self, x: torch.Tensor):
        return torch.sum(
            (x - self.mean_inputs_scaled) @ self.inv_cov_inputs_scaled * (x - self.mean_inputs_scaled),
            dim=1
        )

    def forward(self, x: torch.Tensor):
        ### First, evaluate the network normally
        y = self.net(x)
        y[:, 0] = y[:, 0] - self.squared_mahalanobis_distance(x=x) / (2 * N_inputs)
        ### Add in the squared Mahalanobis distance to the analysis_confidence logit, to ensure it
        # asymptotes to untrustworthy as the inputs get further from the training data

        ### Then, flip the inputs and evaluate the network again.
        # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.

        x_flipped = x.clone()
        x_flipped[:, :8] = -1 * x[:, 8:16]  # switch kulfan_lower with a flipped kulfan_upper
        x_flipped[:, 8:16] = -1 * x[:, :8]  # switch kulfan_upper with a flipped kulfan_lower
        x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
        x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
        x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
        x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

        y_flipped = self.net(x_flipped)
        y_flipped[:, 0] = y_flipped[:, 0] - self.squared_mahalanobis_distance(x=x_flipped) / (2 * N_inputs)
        ### Add in the squared Mahalanobis distance to the analysis_confidence logit, to ensure it
        # asymptotes to untrustworthy as the inputs get further from the training data

        ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
        y_unflipped = y_flipped.clone()
        y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
        y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
        y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
        y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

        # switch upper and lower Ret, H
        y_unflipped[:, 6 + 32 * 0: 6 + 32 * 2] = y_flipped[:, 6 + 32 * 3: 6 + 32 * 5]
        y_unflipped[:, 6 + 32 * 2: 6 + 32 * 3] = y_flipped[:, 6 + 32 * 5: 6 + 32 * 6] * -1  # ue/vinf
        y_unflipped[:, 6 + 32 * 3: 6 + 32 * 5] = y_flipped[:, 6 + 32 * 0: 6 + 32 * 2]
        y_unflipped[:, 6 + 32 * 5: 6 + 32 * 6] = y_flipped[:, 6 + 32 * 2: 6 + 32 * 3] * -1  # ue/vinf

        # switch upper_bl_ue/vinf with lower_bl_ue/vinf

        ### Then, average the two outputs to get the "symmetric" result
        y_fused = (y + y_unflipped) / 2
        # y_fused[:, 0] = torch.sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable
        y_fused[:, 4] = torch.clip(y_fused[:, 4].clone(), 0, 1)  # Top_Xtr clipped to range
        y_fused[:, 5] = torch.clip(y_fused[:, 5].clone(), 0, 1)  # Bot_Xtr clipped to range

        return y_fused

net = Net()

# Lade die Parameter in das Modell
net.load_state_dict(model_state_dict)

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
file_path = "../../data/airfoils/T1_Re0.500_M0.00_N9.0.csv"

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

