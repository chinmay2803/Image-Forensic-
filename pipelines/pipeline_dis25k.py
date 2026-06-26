
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from utils.forensic_helpers import extract_image_forensics_info
# import shared helpers
from utils.forensic_helpers import (
    error_level_analysis,
    noise_residual,
    boundary_artifacts,
    copy_move_detector
)

# ==============================
# CONFIG - update filenames if needed
# ==============================
CLASSIFIER_PATH = "./models/dis25k_classifier.pth"
UNET_PATH = "./models/dis25k_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# thresholds & params (tweakable)
UNET_MASK_THRESHOLD = 0.5
MIN_MASK_RATIO = 0.002      # 0.2% of pixels (for small splices raise lower)
BOUNDARY_SCORE_THRESH = 0.002
ELA_MEAN_THRESH = 6.0
NOISE_VAR_THRESH = 50.0
CMFD_GOOD_MATCHES_THRESH = 30

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
# CLASSIFIER load
# ==============================
classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(f"Classifier file not found: {CLASSIFIER_PATH}")
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE).eval()

# ==============================
# U-NET model (same architecture used for training)
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottom = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.conv1 = DoubleConv(1024,512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.conv2 = DoubleConv(512,256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv3 = DoubleConv(256,128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2);  self.conv4 = DoubleConv(128,64)
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
# Fusion decision logic (dataset-specific thresholds here)
# ==============================
def fuse_decision(class_pred, mask_ratio, boundary_score, ela_mean, noise_var, cm_matches):
    votes = 0.0
    reasons = {}

    # classifier
    if class_pred == 1:
        votes += 1.5
        reasons['classifier'] = True
    else:
        reasons['classifier'] = False

    # segmentation
    if mask_ratio >= MIN_MASK_RATIO:
        votes += 1.5
        reasons['segmentation'] = True
    else:
        reasons['segmentation'] = False

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

    # noise
    if noise_var >= NOISE_VAR_THRESH:
        votes += 0.6
        reasons['noise'] = True
    else:
        reasons['noise'] = False

    # copy-move
    if cm_matches >= CMFD_GOOD_MATCHES_THRESH:
        votes += 1.5
        reasons['copy_move'] = True
    else:
        reasons['copy_move'] = False

    # normalize to confidence 0..1
    max_possible = 1.5 + 1.5 + 1.0 + 0.8 + 0.6 + 1.5  # sum above
    confidence = votes / max_possible

    # choose label
    if confidence >= 0.45:
        # if strong copy-move evidence prefer copy-move label
        if reasons['copy_move']:
            label = "TAMPERED (copy-move likely)"
        elif reasons['segmentation']:
            label = "TAMPERED (splicing likely)"
        else:
            label = "TAMPERED (unspecified)"
    else:
        label = "REAL / No strong evidence"

    return label, float(confidence), reasons

# ==============================
# Main analyze function (single-image) - public API for this pipeline
# ==============================
def analyze_dis25k(image_path, show=False):
    """
    Analyze a single image using DIS25k models and forensic helpers.
    Returns a structured dict with results (suitable for UI).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # PIL image for classifier/unet
    pil = Image.open(image_path).convert("RGB")
    img_np = np.array(pil)

    # classifier (ResNet)
    cls_in = transform_classifier(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cls_output = classifier(cls_in)
        class_pred = int(torch.argmax(cls_output, dim=1).item())

    # unet segmentation
    unet_in = transform_unet(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_pred = unet(unet_in)[0,0].cpu().numpy()
    mask_bin = (mask_pred > UNET_MASK_THRESHOLD).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_area = int(mask_resized.sum())
    mask_ratio = mask_area / float(mask_resized.size) if mask_resized.size > 0 else 0.0

    # overlay (only highlight mask pixels)
    overlay = img_np.copy()
    overlay[mask_resized==1] = [255, 0, 0]  # red for tamper area

    # boundary (uses helper)
    combined, boundary_overlay, boundary_score = boundary_artifacts(image_path, mask_resized)

    # ELA and ELA mean
    ela_img_pil, ela_mean = error_level_analysis(image_path)
    ela_np = np.array(ela_img_pil)

    # noise
    noise_img, noise_var = noise_residual(image_path)

    # copy-move
    cm_img, cm_matches = copy_move_detector(image_path)

    # image metadata + quantization tables (not used in decision but useful info)
    metadata, qtables = extract_image_forensics_info(image_path)
    # fuse decision
    label, confidence, reasons = fuse_decision(
        class_pred, mask_ratio, boundary_score, ela_mean, noise_var, cm_matches
    )

    # optional plotting for local debugging
    if show:
        title = f"Decision: {label}  —  confidence: {confidence:.2f}"
        plt.figure(figsize=(16,10))
        plt.suptitle(title, fontsize=18)

        plt.subplot(2,3,1)
        plt.imshow(img_np); plt.title("Original"); plt.axis("off")

        plt.subplot(2,3,2)
        plt.imshow(mask_resized, cmap='gray'); plt.title(f"U-Net Mask (area {mask_area})"); plt.axis("off")

        plt.subplot(2,3,3)
        plt.imshow(overlay); plt.title("Overlay Heatmap"); plt.axis("off")

        plt.subplot(2,3,4)
        plt.imshow(boundary_overlay); plt.title(f"Boundary Overlay (score {boundary_score:.4f})"); plt.axis("off")

        plt.subplot(2,3,5)
        plt.imshow(ela_np); plt.title(f"ELA (mean {ela_mean:.2f})"); plt.axis("off")

        plt.subplot(2,3,6)
        if cm_img is None:
            plt.imshow(np.zeros_like(img_np)); plt.title("Copy-Move (no features)"); plt.axis("off")
        else:
            plt.imshow(cm_img); plt.title(f"Copy-Move (matches {cm_matches})"); plt.axis("off")

        reason_text = "\n".join([f"{k}: {'YES' if v else 'NO'}" for k,v in reasons.items()])
        reason_text += f"\nnoise_var: {noise_var:.1f}"
        plt.gcf().text(0.02, 0.02, reason_text, fontsize=11, bbox=dict(facecolor='white', alpha=0.75))
        plt.show()

    # return structured info for UI or logging
    return {
    "image": image_path,
    "label": label,
    "confidence": confidence,
    "mask_area": mask_area,
    "mask_ratio": mask_ratio,
    "boundary_score": boundary_score,
    "ela_mean": ela_mean,
    "noise_var": noise_var,
    "cm_matches": cm_matches,
    "reasons": reasons,
    "metadata": metadata,          # ADD THIS LINE
    "qtables": qtables,            # ADD THIS LINE
    "visuals": {
        "original": img_np,
        "mask": (mask_resized * 255).astype(np.uint8),
        "overlay": overlay,
        "boundary_overlay": boundary_overlay,
        "ela": ela_np,
        "copy_move": cm_img
    }
}
    

# ==============================
# CLI quick test
# ==============================
if __name__ == "__main__":
    test_image = "./DIS25k/images/53.jpg"   # change to a valid image path
    res = analyze_dis25k(test_image, show=True)
    print("RESULT:", res)
