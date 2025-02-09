from torch.nn import SELU, SiLU, ReLU
import numpy as np
import matplotlib.pyplot as plt
import torch



x = np.linspace(-10, 10, 10_000)

activation_functions = {
    'SELU': SELU(),
    'SiLU': SiLU(),
    'ReLU': ReLU()
}

plt.figure(figsize=(10, 6))

for name, func in activation_functions.items():
    y = func(torch.tensor(x, dtype=torch.float32)).numpy()
    plt.plot(x, y, label=name)

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
