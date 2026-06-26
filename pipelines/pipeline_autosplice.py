# pipelines/pipeline_autosplice.py
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from utils.forensic_helpers import extract_image_forensics_info

# shared forensic utilities
from utils.forensic_helpers import (
    error_level_analysis,
    boundary_artifacts
)

# ==============================
# CONFIG
# ==============================
CLASSIFIER_PATH = "./models/autosplice_classifier.pth"
UNET_PATH = "./models/autosplice_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# thresholds (tuned for AutoSplice)
UNET_MASK_THRESHOLD = 0.5
BOUNDARY_SCORE_THRESH = 0.002
ELA_MEAN_THRESH = 6.0

# ------------------------------
# TRANSFORMS
# ------------------------------
transform_classifier = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_unet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ==============================
# CLASSIFIER
# ==============================
classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"Classifier file not found: {CLASSIFIER_PATH}")
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE).eval()

# ==============================
# U-NET
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
        u1 = self.conv1(torch.cat([self.up1(btm), d4], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u1), d3], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u2), d2], dim=1))
        u4 = self.conv4(torch.cat([self.up4(u3), d1], dim=1))
        return torch.sigmoid(self.out(u4))

if not os.path.exists(UNET_PATH):
    raise FileNotFoundError(f"U-Net file not found: {UNET_PATH}")
unet = UNet().to(DEVICE)
unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
unet.eval()

# ==============================
# FUSION LOGIC (simpler version)
# ==============================
def fuse_decision(class_pred, boundary_score, ela_mean):
    votes = 0.0
    reasons = {}

    # classifier
    if class_pred == 1:
        votes += 1.5
        reasons['classifier'] = True
    else:
        reasons['classifier'] = False

    # boundary
    if boundary_score >= BOUNDARY_SCORE_THRESH:
        votes += 1.0
        reasons['boundary'] = True
    else:
        reasons['boundary'] = False

    # ELA
    if ela_mean >= ELA_MEAN_THRESH:
        votes += 0.8
        reasons['ELA'] = True
    else:
        reasons['ELA'] = False

    confidence = votes / (1.5 + 1.0 + 0.8)

    if confidence >= 0.4:
        label = "TAMPERED (likely splice)"
    else:
        label = "REAL / No strong evidence"

    return label, float(confidence), reasons

# ==============================
# MAIN ANALYSIS FUNCTION
# ==============================
def analyze_autosplice(image_path, show=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    pil = Image.open(image_path).convert("RGB")
    img_np = np.array(pil)

    # classifier
    cls_input = transform_classifier(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cls_output = classifier(cls_input)
        class_pred = int(torch.argmax(cls_output, dim=1).item())

    # U-Net segmentation
    unet_input = transform_unet(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_pred = unet(unet_input)[0,0].cpu().numpy()
    mask_bin = (mask_pred > UNET_MASK_THRESHOLD).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_area = int(mask_resized.sum())

    overlay = img_np.copy()
    overlay[mask_resized == 1] = [255, 0, 0]

    # boundary + ELA
    _, boundary_overlay, boundary_score = boundary_artifacts(image_path, mask_resized)
    ela_img, ela_mean = error_level_analysis(image_path)
    ela_np = np.array(ela_img)

    # fuse
    label, confidence, reasons = fuse_decision(class_pred, boundary_score, ela_mean)

    metadata, qtables = extract_image_forensics_info(image_path)
    if show:
        title = f"Decision: {label} — confidence: {confidence:.2f}"
        plt.figure(figsize=(14, 9))
        plt.suptitle(title, fontsize=18)

        plt.subplot(2, 3, 1)
        plt.imshow(img_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(mask_resized, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay Heatmap")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(boundary_overlay)
        plt.title(f"Boundary Overlay (score {boundary_score:.4f})")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(ela_np)
        plt.title(f"ELA (mean {ela_mean:.2f})")
        plt.axis("off")

        plt.gcf().text(0.02, 0.02,
                       "\n".join([f"{k}: {'YES' if v else 'NO'}" for k, v in reasons.items()]),
                       fontsize=11,
                       bbox=dict(facecolor='white', alpha=0.7))
        plt.show()

    return {
    "image": image_path,
    "label": label,
    "confidence": confidence,
    "mask_area": mask_area,
    "boundary_score": boundary_score,
    "ela_mean": ela_mean,
    "reasons": reasons,
    "metadata": metadata,          # ADD THIS LINE
    "qtables": qtables,            # ADD THIS LINE
    "visuals": {
        "original": img_np,
        "mask": (mask_resized * 255).astype(np.uint8),
        "overlay": overlay,
        "boundary_overlay": boundary_overlay,
        "ela": ela_np
    }
}

# ==============================
# CLI quick test
# ==============================
if __name__ == "__main__":
    test_image = "./Forged_JPEG90/71695_0.jpg"
    res = analyze_autosplice(test_image, show=True)
    print("RESULT:", res)
