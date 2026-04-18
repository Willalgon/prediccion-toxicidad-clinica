# 1. Traducción de química a álgebra
# 2. Batching
# 3. Tensores

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # utilizaremos float32 bits para torch
import torch
from torch.utils.data import Dataset, DataLoader # Dataset (interfaz para heredar len y getitem) # DataLoader para cargar los batches
from sklearn.model_selection import train_test_split 
from rdkit import Chem # Chem entiende la química de los SMILES
from rdkit import rdMolDescriptors # Para generar los vectores Morgan Fingerprint


class ClinToxDataset(Dataset): # obtenemos los datos, longitud, y obtenemos las muestras ya vectorizadas
    def __init__(self, dataframe):
        self.data = dataframe
        self.smiles = self.data["smiles"].values
        self.labels = self.data["CT_TOX"].values


    def __len__(self):
        return len(self.labels)

    def smiles_to_fingerprint(self, smiles_str):
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return np.zeros(2048, dtype=np.float32)
        # Morgan Fingerprint
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect( mol, radius=2, nBits=2048)
        return np.array(fingerprint, dtype=np.float32)

    def __getitem__(self, id): # 
        smile_str = self.smiles[id]
        label = self.labels[id]
        # Transformamos la química en un vector
        features = self.smiles_to_fingerprint(smile_str)
        # Convertimos a tensores
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.float32)
        return x_tensor, y_tensor


def get_dataloaders(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)
    df = df.dropna() # para eliminar de las columnas sibset las filas sin contenido
    # Dividimos
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["CT_TOX"])
    # Queremos el mismo random state siempre para guardar la distribución y no cambiar los resultados de las métricas
    # Obtenemos conjunto de validación
    train_df, val_df = train_test_split( train_val_df, test_size=0.15, random_state=42, stratify=train_val_df["CT_TOX"])

    train_ds = ClinToxDataset(train_df)
    val_ds = ClinToxDataset(val_df)
    test_ds = ClinToxDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(
        f"Dataset is ready: {len(train_df)} Train, {len(val_df)} Validation, {len(test_df)} Test"
    )
    return train_loader, val_loader, test_loader
