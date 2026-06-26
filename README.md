# 🔬 AutoSplice — AI-Powered Image Forgery Detection

AutoSplice is a web-based digital image forensics tool that detects whether an image has been **digitally tampered with** — for example through splicing, copy-move forgery, or object insertion. It combines deep learning models with classic forensic image-analysis techniques, and presents the results through a clean web dashboard along with a downloadable PDF forensic report.

The system doesn't just say *"real"* or *"fake"* — it shows **why**, by visualizing tampered regions, highlighting suspicious boundaries, and surfacing image metadata that a human investigator would also look at.

---

## ✨ Key Features

- 🧠 **Dual detection pipelines** — choose between two independently trained model sets (`AutoSplice` and `DIS25k`), or run both together for a combined verdict.
- 🩻 **Pixel-level localization** — a U-Net segmentation model highlights exactly *which region* of the image was tampered with, not just a yes/no label.
- 🔍 **Multi-technique forensic fusion** — combines a CNN classifier, segmentation mask, boundary-artifact detection, Error Level Analysis (ELA), noise residual analysis, and copy-move detection into one weighted confidence score.
- 📊 **Metadata & EXIF inspection** — extracts EXIF data and JPEG quantization tables, which often reveal re-compression or editing history.
- 📄 **One-click PDF report** — generates a downloadable, presentation-ready forensic report with all visuals, scores, and metadata.
- 🎨 **Modern web UI** — a simple drag-and-drop Flask interface, no command-line knowledge needed to use it.

---

## 🧩 How It Works

Every uploaded image is run through a **fusion pipeline** that gathers independent pieces of forensic evidence and combines them into a single confidence score. No single technique is trusted blindly — if one signal is wrong, the others act as a safety net.

| Technique | What it does | Why it matters |
|---|---|---|
| **CNN Classifier** (ResNet18) | A binary classifier that gives a fast first opinion: *Authentic* or *Forged*. | Quick triage step — but never the only word, since classifiers can be wrong. |
| **U-Net Segmentation** | Predicts a pixel-level mask of which exact region was tampered with. | Turns a vague "this is fake" into "this *specific area* is fake." |
| **Overlay Heatmap** | Paints the predicted tampered region directly on top of the original image. | Gives a clear, visual proof an investigator can point to. |
| **Boundary Artifact Detection** (Laplacian + Sobel) | Looks for unnatural edges around objects — the kind of sharp, inconsistent boundary that copy-pasted objects often leave behind. | Real photo edges are usually smooth and consistent; spliced objects often aren't. |
| **Error Level Analysis (ELA)** | Re-compresses the image as JPEG and measures the difference from the original. | Tampered regions are re-compressed differently than the rest of the photo, so they "light up" in the ELA map. |
| **Noise Residual Analysis** | Extracts the image's underlying noise pattern using a Laplacian filter. | Genuine photos have a consistent noise/grain pattern across the whole frame; pasted-in regions usually don't match it. |
| **Copy-Move Detection** (ORB feature matching) | Searches the image for regions that are duplicates of each other. | Catches "clone stamp" style forgeries, where part of an image is copied and pasted elsewhere in the same image. |
| **EXIF & Quantization Table Analysis** | Reads embedded camera metadata and JPEG compression tables. | Missing EXIF data or inconsistent quantization tables are classic fingerprints of edited images. |

Each technique casts a weighted "vote." The votes are summed into a confidence score, and a label (e.g. `TAMPERED (splicing likely)`, `TAMPERED (copy-move likely)`, or `REAL / No strong evidence`) is produced based on that score.

### The Two Pipelines

AutoSplice ships with **two separate model pairs** (a classifier + a U-Net each), trained on two different datasets:

- **AutoSplice pipeline** — trained on the AutoSplice dataset (authentic vs. JPEG-recompressed forged images). Uses the classifier, U-Net mask, boundary detection, and ELA.
- **DIS25k pipeline** — trained on the DIS25k dataset. Adds noise residual analysis and copy-move detection on top of the same core techniques, making it the more thorough of the two.

You can run either one individually from the UI, or select **"Both — Comprehensive Scan"** to run them side by side and compare verdicts.

---

## 🏗️ Project Structure

```
AutoSplice/
├── app/
│   ├── app.py                  # Flask application & routes
│   ├── static/
│   │   ├── css/style.css       # Frontend styling
│   │   └── uploads/            # Uploaded images & generated visuals (runtime)
│   └── templates/
│       ├── index.html          # Upload page
│       └── result.html         # Results / report page
│
├── pipelines/
│   ├── pipeline_manager.py     # Unified entry point — routes to the right pipeline
│   ├── pipeline_autosplice.py  # AutoSplice model pipeline
│   ├── pipeline_dis25k.py      # DIS25k model pipeline
│   ├── test_pipeline.py        # Quick CLI test for the AutoSplice pipeline
│   └── test_pipeline_dis25k.py # Quick CLI test for the DIS25k pipeline
│
├── utils/
│   ├── forensic_helpers.py     # ELA, noise, boundary, copy-move, EXIF/qtable helpers
│   └── pdf_generator.py        # Builds the downloadable PDF forensic report
│
├── training_scripts/
│   ├── train_classifier.py         # Trains the AutoSplice ResNet18 classifier
│   ├── train_unet.py               # Trains the AutoSplice U-Net segmentation model
│   ├── train_dis25k_classifier.py  # Trains the DIS25k ResNet18 classifier
│   └── train_dis25k_unet.py        # Trains the DIS25k U-Net segmentation model
│
├── techniques.md               # Notes on each forensic technique used
├── requirements.txt
└── README.md
```

> **Note on the `models/` folder:** The trained `.pth` model weights are **not included in this repository** (they're large binary files and excluded via `.gitignore`). See [Training Your Own Models](#-training-your-own-models) below to generate them yourself.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend / Web | Flask, Werkzeug |
| Deep Learning | PyTorch, Torchvision (ResNet18, custom U-Net) |
| Image Processing | OpenCV, Pillow, NumPy |
| Reporting | ReportLab (PDF generation) |
| Training Utilities | Pandas, scikit-learn, tqdm |
| Frontend | HTML, TailwindCSS (CDN), vanilla JS |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chinmay2803/Image-Forensic-.git
cd Image-Forensic-/AutoSplice
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you have an NVIDIA GPU and want CUDA acceleration, install a CUDA-enabled build of PyTorch from the [official PyTorch website](https://pytorch.org/get-started/locally/) **before** running the command above, so it doesn't get overwritten by the CPU-only build.

### 4. Add the trained models

Place the four trained model files inside a `models/` folder in the project root (this folder is git-ignored and must be created manually):

```
models/
├── autosplice_classifier.pth
├── autosplice_unet.pth
├── dis25k_classifier.pth
└── dis25k_unet.pth
```

Don't have them yet? See [Training Your Own Models](#-training-your-own-models) below.

### 5. Run the app

From the project **root directory**:

```bash
python -m app.app
```

Then open **http://127.0.0.1:5000** in your browser, upload an image, pick a detection mode, and view the results.

---

## 🎓 Training Your Own Models

The `training_scripts/` folder contains the scripts originally used to train all four models. They are provided for reference and reproducibility — you'll need to point them at your own copies of the **AutoSplice** and **DIS25k** datasets.

| Script | Trains | Output |
|---|---|---|
| `train_classifier.py` | AutoSplice ResNet18 classifier | `autosplice_classifier.pth` |
| `train_unet.py` | AutoSplice U-Net segmentation model | `autosplice_unet.pth` |
| `train_dis25k_classifier.py` | DIS25k ResNet18 classifier | `dis25k_classifier.pth` |
| `train_dis25k_unet.py` | DIS25k U-Net segmentation model | `dis25k_unet.pth` |

**Before running any script**, open it and update the `DATA_DIR` (and related path variables) at the top of the file to point to where your dataset is stored on disk — they currently contain the original author's local paths as placeholders.

```bash
python training_scripts/train_classifier.py
python training_scripts/train_unet.py
python training_scripts/train_dis25k_classifier.py
python training_scripts/train_dis25k_unet.py
```

Once training completes, move the resulting `.pth` files into the `models/` folder described in step 4 above.

---

## 📄 PDF Forensic Report

After any analysis, click **"Download PDF Report"** to generate a shareable report containing:
- The final verdict and confidence-driving techniques
- Image metadata and EXIF / quantization table details
- All forensic visualizations (mask, overlay, boundary map, ELA, noise, copy-move) on a dedicated page

This is handy for documentation, academic submissions, or sharing findings with someone who doesn't have access to the app.

---

## 🛣️ Possible Future Improvements

- Batch processing for multiple images at once
- REST API endpoints for programmatic access
- Support for additional forgery types (e.g. AI-generated image / deepfake detection)
- Dockerized deployment for easier setup

---

## 👤 Author

**Chinmay Bolinjkar**
Final-year Computer Engineering student, VCET | Android Developer | ML & Forensics Enthusiast
