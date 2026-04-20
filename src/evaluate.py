import torch
import torch.nn as nn
from data_loader import get_dataloaders
from model import DeepToxModel
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate():
    print("Evaluating Best Model")
    batch_size = 32
    _, _, test_loader = get_dataloaders("data/clintox.csv", batch_size=batch_size)
    # instanciar modelo y cargar los mejores parámetros
    model = DeepToxModel()
    model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.15).float() # Umbral de decisión al 50%
            # llevamos a cpu y convertimos a numpy
            all_probs.extend(probs.cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Cálculo de métricas
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=["Non-Toxic", "Toxic"]))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Non-Toxic", "Toxic"], 
                yticklabels=["Non-Toxic", "Toxic"])
    plt.title('Confusion Matrix - DeepTox')
    plt.ylabel('Real')
    plt.xlabel('Predicted')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    evaluate()