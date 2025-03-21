import torch
import torch.nn as nn

from core.utils.registry import registry

@registry.register_model("mlp")
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
