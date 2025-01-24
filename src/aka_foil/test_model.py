from airfoil_handling_functions import *
import torch

# Paths
airfoil_path = '../../data/airfoils/acc22.dat'
model_save_path = '../../data/train_aka_foil_2025-01-17_13-11-02/' \
                  'train_aka_foil_1f371_00003_3_activation_fn=ref_ph_d09f660c,hidden_sizes=64,lr=0.0070,n_hidden_layers=5_2025-01-17_13-11-03/' \
                  'checkpoint_000499/checkpoint.pt'

# Operating point
alpha = 10
Re = 1e6
mach = 0
n_crit = 9
xtr_upper = 1
xtr_lower = 1

# Transform Airfoil and Operating Point into input format
kulfan_parameters = get_kulfan_parameters(get_file_coordinates(airfoil_path))

lower_weights = torch.tensor(kulfan_parameters['lower_weights'], dtype=torch.float32)
upper_weights = torch.tensor(kulfan_parameters['upper_weights'], dtype=torch.float32)
te_thickness = torch.tensor([kulfan_parameters['TE_thickness']], dtype=torch.float32)
leading_edge_weight = torch.tensor([kulfan_parameters['leading_edge_weight']], dtype=torch.float32)
alpha = torch.tensor([alpha], dtype=torch.float32)
Re = torch.tensor([Re], dtype=torch.float32)
mach = torch.tensor([mach], dtype=torch.float32)
n_crit = torch.tensor([n_crit], dtype=torch.float32)
xtr_upper = torch.tensor([xtr_upper], dtype=torch.float32)
xtr_lower = torch.tensor([xtr_lower], dtype=torch.float32)

input = torch.cat([lower_weights, upper_weights, leading_edge_weight, te_thickness,
                   alpha, Re, mach, n_crit, xtr_upper, xtr_lower])

# Reload model and evaluate for input
model = torch.load(model_save_path)
model.eval()
output = model(input)
print(output)
