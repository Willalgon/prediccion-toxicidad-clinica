import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepToxModel(nn.Module):
    def __init__(self, input_size=2048):
        super().__init__()
        # Definimos las capas fully connected:
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 500)
        self.fc3 = nn.Linear(500, 260)
        self.out = nn.Linear(260, 1) # Binaria, predicción
        # definimos un dropout para apagar neuronas al azar para evitar overfitting
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.fc1(x) # z1 = x Wt + b1
        x = F.relu(x)   # a1 = max(0, z1)
        x = self.dropout(x) # dropout a alguna neurona

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x) # z3 -> logit
        # No aplicamos reLU ya que BCE utiliza los logits

        return x  