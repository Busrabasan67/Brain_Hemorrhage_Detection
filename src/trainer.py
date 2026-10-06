import os
import copy
import torch
import pandas as pd
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, output_dir, label_smoothing=False):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, 'en_iyi_model.pth')
    csv_path = os.path.join(output_dir, 'egitim_gecmisi.csv')

    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        y_true_train, y_pred_train = [], []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            
            if label_smoothing:
                # e.g., 1.0 -> 0.95, 0.0 -> 0.05
                smoothed_labels = labels.float() * 0.9 + 0.05
                target = smoothed_labels.to(device).unsqueeze(1)
            else:
                target = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            y_pred_train.extend((probs > 0.5).astype(int))
            y_true_train.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(y_true_train, y_pred_train)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_pred_val.extend((probs > 0.5).astype(int))
                y_true_val.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(y_true_val, y_pred_val)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # En iyi modeli kaydet (Öncelik: En yüksek Validasyon Doğruluğu, Eşitlikte: En düşük Loss)
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, weights_path)
            print(f">>> Yeni en iyi model! (Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f}) -> {weights_path} adresine kaydedildi.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"!!! Early Stopping tetiklendi. {patience} epoch boyunca gelişme olmadı.")
                break

    # Save History CSV
    pd.DataFrame(history).to_csv(csv_path, index=False)
    print(f"✅ Eğitim geçmişi '{csv_path}' olarak kaydedildi.")
    
    # Load best weights before returning
    model.load_state_dict(best_model_weights)
    return history, model
