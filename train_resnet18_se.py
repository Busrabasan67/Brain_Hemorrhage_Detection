import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import split_data, BrainCTDataset, get_transforms
from src.models.resnet18_se import get_resnet18_se
from src.trainer import train_model
from src.utils import evaluate_model, plot_history

# Ayarlar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')
CSV_FILE = os.path.join(BASE_DIR, 'data', 'labels.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'resnet18_se')

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- EĞİTİM BAŞLIYOR: ResNet18 + SE-Block ---")
    print(f"Cihaz: {DEVICE}")

    # 1. Veriyi Böl
    train_df, val_df, test_df = split_data(CSV_FILE, DATA_DIR)

    # 2. Transformları al
    train_transforms, val_test_transforms = get_transforms(is_v6=False)

    # 3. Dataset ve DataLoader oluştur
    train_dataset = BrainCTDataset(train_df, DATA_DIR, transform=train_transforms)
    val_dataset = BrainCTDataset(val_df, DATA_DIR, transform=val_test_transforms)
    test_dataset = BrainCTDataset(test_df, DATA_DIR, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Modeli Oluştur ve Ayarla
    model = get_resnet18_se(num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Eğitimi Başlat
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None, # scheduler yok
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        output_dir=OUTPUT_DIR,
        label_smoothing=False
    )

    # 6. Sonuçları Çizdir
    plot_history(history, OUTPUT_DIR)

    # 7. Test Et
    evaluate_model(best_model, test_loader, DEVICE, OUTPUT_DIR)

if __name__ == '__main__':
    main()
