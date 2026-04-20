import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepToxModel(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        # Definimos las capas fully connected:
        self.fc1 = nn.Linear(input_size, 150)
        self.fc2 = nn.Linear(150, 40)
        self.out = nn.Linear(40, 1) # Binaria, predicción
        # definimos un dropout para apagar neuronas al azar para evitar overfitting
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = self.fc1(x) # z1 = x Wt + b1
        x = F.tanh(x)   # a1 = max(0, z1)
        x = self.dropout(x) # dropout a alguna neurona

        x = self.fc2(x)
        x = F.tanh(x)
        x = self.dropout(x)

        x = self.out(x) # z3 -> logit
        # No aplicamos reLU ya que BCE utiliza los logits

        return x  