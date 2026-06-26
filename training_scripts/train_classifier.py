import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_DIR = r"C:\Users\rboli\Downloads\AutoSplice"
TRAIN_DIR = os.path.join(DATA_DIR, "ClassifierData")  # clean dataset dir
MODEL_PATH = "autosplice_classifier.pth"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {DEVICE}")

# ==============================
# PREPARE CLEAN DATASET
# ==============================
def copy_folder(src, dst):
    if os.path.exists(dst):
        return  # already copied
    shutil.copytree(src, dst)

os.makedirs(TRAIN_DIR, exist_ok=True)

copy_folder(os.path.join(DATA_DIR, "Authentic"), os.path.join(TRAIN_DIR, "Authentic"))
copy_folder(os.path.join(DATA_DIR, "Forged_All"), os.path.join(TRAIN_DIR, "Forged_All"))

print("✅ Dataset structure ready:", os.listdir(TRAIN_DIR))

# ==============================
# DATASET
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
print("✅ Classes detected:", dataset.classes)
print(f"✅ Dataset ready: {len(dataset)} images total")

# Split into train/val (80/20)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================
# MODEL
# ==============================
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # binary classifier

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==============================
# TRAINING
# ==============================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"📊 Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Saved best model")

print("🎉 Training complete. Best Val Acc:", best_val_acc)
