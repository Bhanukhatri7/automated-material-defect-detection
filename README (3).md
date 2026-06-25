# Automated Material Defect Detection using CNN Architectures

## 📌 Project Overview

Surface micro-crack detection in structural and composite materials is a critical quality control challenge. Manual visual inspection introduces grading bias, inconsistency, and processing delays at scale. This project implements a custom Convolutional Neural Network (CNN) using PyTorch to fully automate binary classification of surface images into cracked vs. clean categories.

**Dataset:** Concrete Crack Images for Classification (Özgenel, Çağlar Fırat — Mendeley Data, 2019)  
**Source:** [Kaggle — Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)  
**40,000 real surface images** — 20,000 positive (cracked), 20,000 negative (clean). Perfectly balanced binary classification task.

---

## 🚀 Key Results

| Metric | Value |
|---|---|
| Test Accuracy | 99.78% |
| Precision (Crack class) | 99.55% |
| Recall (Crack class) | 99.5% |
| Misclassifications | 18 / 8,000 test images |
| Total Dataset Size | 40,000 images |
| Train / Test Split | 80 / 20 |

---

## 🧠 Methodology

### 1. Dataset
40,000 real surface photographs (20,000 cracked, 20,000 clean) sourced from Özgenel's publicly available concrete crack dataset. The dataset is perfectly balanced — no class imbalance adjustment or oversampling was required. Each image captures a unique surface region under varying lighting and angle conditions.

### 2. Preprocessing Pipeline
Raw images were standardised through a fixed `torchvision.transforms` pipeline applied consistently to both train and test sets:
- **Resize** → 128×128 pixels
- **Grayscale conversion** → RGB (3-channel) to single-channel, reducing input dimensionality and computational load
- **Normalisation** → pixel values scaled before tensor conversion

### 3. Model Architecture
Custom multi-layer CNN built from scratch in PyTorch:
- `Conv2d` layers for edge detection and micro-crack pattern extraction
- `MaxPooling2d` for spatial dimensionality reduction
- `ReLU` activation functions throughout
- Fully connected output layer → 2 classes (cracked / clean)

### 4. Training Configuration
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Split:** 80/20 stratified train-test split
- **Test set size:** 8,000 unseen images

---

## 📊 Results & Confusion Matrix

The model was evaluated on 8,000 held-out test images never seen during training. The confusion matrix below confirms a minimal false-negative rate — critical in quality control contexts where missing a crack is more costly than a false alarm.

![Confusion Matrix](confusion_matrix.png)

Replacing subjective human visual grading with deterministic model predictions eliminated inter-grader variability entirely from the classification pipeline.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML Framework | PyTorch, Torchvision |
| Data & Visualisation | NumPy, Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Environment | Jupyter Notebook |

---

## 📁 Repository Structure

```
├── defect_detection.ipynb    # Full training pipeline and evaluation
├── confusion_matrix.png      # Model output — confusion matrix
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

---

## 📚 Dataset Citation

Özgenel, Çağlar Fırat (2019), *Concrete Crack Images for Classification*, Mendeley Data, V2.  
DOI: [10.17632/5y9wdsg2zt.2](https://doi.org/10.17632/5y9wdsg2zt.2)
