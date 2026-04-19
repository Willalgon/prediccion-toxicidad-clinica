import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders
from model import DeepToxModel
import os
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit import Chem
import pandas as pd

def train():
    # Tenemos que limpiar las moléculas erróneas...
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    print("Cleaning dataset (erroneous molecules)")
    df_raw = pd.read_csv("data/clintox.csv")
    # Lista donde guardaremos solo las filas que pasen el filtro
    valid_rows = []
    for _, row in df_raw.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        # si la molécula es válida la guardamos
        if mol is not None:
            valid_rows.append(row)
    # Creamos el nuevo DataFrame con las filas filtradas
    df_clean = pd.DataFrame(valid_rows).copy()
    # Guardamos el temporal para que el DataLoader lo use
    clean_path = "data/clintox_cleaned.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"Filtered dataset: {len(df_raw) - len(df_clean)} molecules removed")
    print("------------------------------------------")

    # -----  ENTRENAMIENTO ------

    # Hiperparámetros
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 10 # épocas que esperamos si no mejora la validación
    # cargamos datos y modelo
    train_loader, val_loader, _ = get_dataloaders(clean_path, batch_size=batch_size)
    model = DeepToxModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Variables para early stopping
    best_val_loss = float('inf') # infinito positivo
    epochs_no_improve = 0

    # Listas para guardar la historia
    history_train_loss = []
    history_val_loss = []

    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad() # eliminamos pendientes anteriores
            outputs = model(batch_x).squeeze() # forward y eliminamos dimensión 1 del tensor de salida.

            loss = criterion(outputs, batch_y)
            loss.backward() # guardamos gradientes en model.parameters()
            optimizer.step() # ajustamos los pesos y bias con Adam

            # como loss no es una variable sino un tensor de un solo elemento
            # extraemos ese valor float
            train_loss += loss.item()

        # Validación
        model.eval()
        val_loss = 0.0
        # obviamente ya no queremos backward
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        # Métricas de la época
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # guardamos para graficar
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        print(f"Epoch [{epoch+1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Eraly stopping y guardado del mejor modelo:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # guardamos los mejores pesos hasat el momento. Sobreescribe
            torch.save(model.state_dict(), "models/best_model.pth")
        
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping launched. The model does not improve since {patience} epochs.")
                break

    # generamos la gráfica al  terminar el entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(history_train_loss, label='Train Loss', color='blue')
    plt.plot(history_val_loss, label='Validation Loss', color='orange')
    
    # línea roja donde se guardó el mejor modelo
    best_epoch = len(history_train_loss) - epochs_no_improve - 1
    plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Model (Epoch {best_epoch+1})')
    
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/learning_curves.png')
    plt.show()

if __name__ == "__main__":
    train()

