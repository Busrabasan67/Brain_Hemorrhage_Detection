import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import torch

def evaluate_model(model, test_loader, device, output_dir=None):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(predicted)
            y_probs.extend(probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- TEST SET RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    if output_dir is not None:
        # Metrikleri JSON'a Kaydetme
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics_dict = {
            "accuracy": f"{acc*100:.1f}%",
            "precision": f"{prec*100:.1f}%",
            "recall": f"{rec*100:.1f}%",
            "f1": f"{f1_score*100:.1f}%"
        }
        with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print("Model metrikleri 'test_metrics.json' dosyasına kaydedildi.")

        # Confusion Matrix PNG Kaydetme
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy (0)', 'Hemorrhage (1)'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion Matrix (PNG) saved to: {cm_path}")
        plt.close()
        
        # ROC Eğrisi Kaydetme
        try:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_path)
            print(f"ROC Curve (PNG) saved to: {roc_path}")
            plt.close()
        except Exception as e:
            print(f"ROC couldn't be plotted: {str(e)}")
    
    return acc, prec, rec, cm

def plot_history(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_graphs.png')
    plt.savefig(plot_path)
    print(f"Graphs saved to {plot_path}")
    plt.close()

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
