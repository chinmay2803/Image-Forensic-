## 1. Classifier (ResNet18 – Binary Classification)
Kya karta hai:
Pre-trained ResNet18 (modified last layer for 2 classes) lagaya gaya hai.
Ye ek binary classifier hai jo image ko Authentic (Real) ya Forged (Tampered) label deta hai.

Use case:

Quick classification step, fast check.

Lekin agar classifier galat bhi bole, hum forensic proofs le aate hain baaki techniques se (failsafe approach).

## 2. U-Net (Segmentation)
Kya karta hai:
U-Net ek segmentation model hai jo image ke har pixel ke liye decide karta hai ki ye tampered hai ya nahi.
Iska output ek binary mask hota hai (0 = authentic pixel, 1 = tampered pixel).

Use case:

Tampered region localize karna.

Matlab fake area ka exact region highlight hota hai (for example sticker add, object clone, etc.).

## 3. Overlay Heatmap
Kya karta hai:
U-Net ke predicted mask ko original image ke upar overlay kar deta hai.
Tampered regions ko red highlight diya jaata hai.

Use case:

Visual forensic proof: tumhe clearly dikhega ki kahan manipulation hua hai.

## 4. Boundary Artifact Detector (Laplacian + Sobel Edges)
Kya karta hai:

Laplacian filter → edges aur sharp changes detect karta hai.

Sobel filter → horizontal + vertical gradients detect karta hai.

Dono ko combine karke suspicious boundaries highlight ki jaati hain.

Mask ke andar restrict:
Agar U-Net mask available hai, toh boundary highlights sirf tampered region ke andar dikhega (pure image nahi).

Use case:
Fake stickers, copy-paste objects, ya splicing mein edge boundaries abnormal hoti hain → ye unko highlight karega.

## 5. Error Level Analysis (ELA)
Kya karta hai:

Image ko phirse JPEG format mein compress karke uski difference nikalta hai (original - recompressed).

Authentic regions usually ek jaisa compress hote hain, lekin tampered regions mein compression mismatch hota hai.

Use case:
Ye technique tab kaam aati hai jab JPEG image manipulate kiya gaya ho.
Tampered area ELA mein zyada bright/different dikhega.

## 6. Noise Residual Analysis
Kya karta hai:

Laplacian filter use karke image ka noise pattern extract karta hai.

Authentic image ke noise usually uniform hota hai,
lekin tampered areas mein noise pattern mismatch hota hai (alag intensity, alag texture).

Use case:
Photoshop/Picsart jaise edits mein sticker ya clone ka noise fingerprint original se alag hota hai.