import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import DeepToxModel
import os

def train():
    # Hiperparámetros
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patiente = 10 # épocas que esperamos si no mejora la validación
    # cargamos datos y modelo
    train_loader, val_loader, _ = get_dataloaders("data/clintox.csv", batch_size=batch_size)
    model = DeepToxModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Variables para early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True) # en esta carpeta guardaremos el modelo
    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()   
            outputs = model(batch_x).squeeze()

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch


