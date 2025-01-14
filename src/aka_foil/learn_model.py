import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers, hidden_size):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.SELU())
        
        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SELU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Example usage
if __name__ == "__main__":
    input_size = 10
    output_size = 2
    n_hidden_layers = 3
    hidden_size = 50
    
    model = MLP(input_size, output_size, n_hidden_layers, hidden_size)
    print(model)
