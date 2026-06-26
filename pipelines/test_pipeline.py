import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image, ImageChops, ImageEnhance

# ==============================
# CONFIG
# ==============================
CLASSIFIER_PATH = "./models/autosplice_classifier.pth"
UNET_PATH = "./models/autosplice_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# TRANSFORMS
# ==============================
transform_classifier = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_unet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ==============================
# CLASSIFIER (ResNet18)
# ==============================
classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE)
classifier.eval()

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

        u1 = self.up1(btm)
        u1 = self.conv1(torch.cat([u1, d4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, d3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, d2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv4(torch.cat([u4, d1], dim=1))

        return torch.sigmoid(self.out(u4))

unet = UNet().to(DEVICE)
unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
unet.eval()

# ==============================
# EXTRA TECHNIQUES
# ==============================
def error_level_analysis(img_path):
    """ Perform ELA on JPEG """
    orig = Image.open(img_path).convert("RGB")
    resaved = "temp_resaved.jpg"
    orig.save(resaved, "JPEG", quality=90)
    resaved_img = Image.open(resaved)

    diff = ImageChops.difference(orig, resaved_img)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    return diff



def boundary_artifact_detector(image_path, mask_resized=None):
    """Detect suspicious boundaries using Laplacian + Sobel and restrict to mask if available."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(255 * sobel / np.max(sobel))

    combined = cv2.addWeighted(laplacian, 0.5, sobel, 0.5, 0)

    if mask_resized is not None:
        combined = cv2.bitwise_and(combined, combined, mask=mask_resized.astype(np.uint8))

    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 0.7, heatmap, 0.3, 0)

    return combined, overlay

# ==============================
# FAILSAFE PIPELINE
# ==============================
def test_pipeline(image_path):
    img = Image.open(image_path).convert("RGB")

    # --- Classifier (only for label info) ---
    cls_input = transform_classifier(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cls_output = classifier(cls_input)
        pred_class = torch.argmax(cls_output, dim=1).item()

    if pred_class == 0:
        print("Classifier says: ✅ Authentic (Real)")
    else:
        print("Classifier says: ⚠️ Forged (Tampered)")

    # --- U-Net Mask ---
    unet_input = transform_unet(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_pred = unet(unet_input)[0,0].cpu().numpy()
    mask_pred = (mask_pred > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_pred, img.size, interpolation=cv2.INTER_NEAREST)

    overlay = np.array(img).copy()
    overlay[mask_resized==1] = [255,0,0]

    # --- Boundary Artifacts ---
    boundary_map, boundary_overlay = boundary_artifact_detector(image_path, mask_resized)

    # --- ELA ---
    ela_img = error_level_analysis(image_path)


   

    # --- Show results ---
    plt.figure(figsize=(16,10))

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(mask_resized, cmap="gray")
    plt.title("Predicted Mask (U-Net)")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(overlay)
    plt.title("Overlay Heatmap")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(boundary_map, cmap="gray")
    plt.title("Boundary Map")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(boundary_overlay)
    plt.title("Boundary Overlay")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.imshow(ela_img)
    plt.title("Error Level Analysis (ELA)")
    plt.axis("off")

    plt.show()

    

# ==============================
# RUN TEST
# ==============================
if __name__ == "__main__":
    test_image = "./Forged_JPEG90/71695_0.jpg"
    test_pipeline(test_image)






#  ./Forged_JPEG90/71695_0.jpg