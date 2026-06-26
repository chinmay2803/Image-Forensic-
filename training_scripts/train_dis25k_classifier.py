import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_DIR = r"C:\Users\rboli\Downloads\AutoSplice\DIS25k"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "metaDataTrain.csv")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MODEL_PATH = "dis25k_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {DEVICE}")

# ==============================
# DATASET
# ==============================
class DIS25kDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["imageId"]
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")

        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    # Load metadata
    df = pd.read_csv(CSV_PATH)

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])

    train_dataset = DIS25kDataset(train_df, transform=transform)
    val_dataset = DIS25kDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Dataset ready: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # ==============================
    # MODEL
    # ==============================
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ==============================
    # TRAINING LOOP
    # ==============================
    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        running_loss, correct = 0.0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / len(train_dataset)

        # ---- Validation ----
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_dataset)

        print(f"📊 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), MODEL_PATH)

    print("🎉 Training complete. Model saved to", MODEL_PATH)
