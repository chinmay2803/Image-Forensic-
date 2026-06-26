import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_DIR = r"C:\Users\rboli\Downloads\AutoSplice\DIS25k"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "groundtruths")
CSV_PATH = os.path.join(DATA_DIR, "metaDataTrain.csv")

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
PATIENCE = 5
MODEL_PATH = "dis25k_unet.pth"
CHECKPOINT_PATH = "dis25k_unet_checkpoint.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {DEVICE}")

# ==============================
# DATASET
# ==============================
class DIS25kSegDataset(Dataset):
    def __init__(self, df, transform_img=None, transform_mask=None):
        self.df = df
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["imageId"]

        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
        mask_path = os.path.join(MASK_DIR, f"{img_id}.jpg")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return img, mask

# ==============================
# TRANSFORMS
# ==============================
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==============================
# U-NET MODEL
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(nn.MaxPool2d(2)(d1))
        d3 = self.down3(nn.MaxPool2d(2)(d2))
        d4 = self.down4(nn.MaxPool2d(2)(d3))
        btm = self.bottom(nn.MaxPool2d(2)(d4))

        u1 = self.up1(btm)
        u1 = self.conv1(torch.cat([u1, d4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, d3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, d2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv4(torch.cat([u4, d1], dim=1))

        return torch.sigmoid(self.out(u4))

# ==============================
# TRAINING with CHECKPOINT
# ==============================
if __name__ == "__main__":
    # Load metadata
    df = pd.read_csv(CSV_PATH)

    # Split train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_dataset = DIS25kSegDataset(train_df, transform_img, transform_mask)
    val_dataset = DIS25kSegDataset(val_df, transform_img, transform_mask)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Dataset ready: Train={len(train_dataset)}, Val={len(val_dataset)}")

    model = UNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    start_epoch = 0
    epochs_no_improve = 0

    # ✅ Resume if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        print("♻️ Resuming training from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        print(f"🔁 Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            masks = (masks > 0.5).float()

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                masks = (masks > 0.5).float()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"📊 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- Save checkpoint ----
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Saved best U-Net model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("⏹ Early stopping triggered")
                break

    print("🎉 Training complete. Best Val Loss:", best_val_loss)
