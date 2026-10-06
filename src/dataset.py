import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def split_data(csv_path, data_dir):
    print("CSV okunuyor ve resimlerin sağlamlığı kontrol ediliyor...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['filename'] = df['id'].astype(str).str.zfill(3) + '.png'

    gecerli_dosyalar = []
    for dosya_adi in df['filename']:
        dosya_yolu = os.path.join(data_dir, dosya_adi)
        if os.path.exists(dosya_yolu):
            img = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
            gecerli_dosyalar.append(img is not None)
        else:
            gecerli_dosyalar.append(False)

    df = df[gecerli_dosyalar].reset_index(drop=True)
    
    # %70 Train, %15 Val, %15 Test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['hemorrhage'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['hemorrhage'])
    
    return train_df, val_df, test_df

class BrainCTDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, apply_clahe=True, channels=3):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.channels = channels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        if self.channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        image = Image.fromarray(image)
        label = float(self.dataframe.loc[idx, 'hemorrhage'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(is_v6=False):
    if is_v6:
        train_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    val_test_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_t, val_test_t
