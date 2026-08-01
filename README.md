# CNN Image Classification Model — Acne vs Eczema (CO9 Lab 10)

A binary image classifier that distinguishes **Acne** from **Eczema** skin conditions using transfer learning with **MobileNetV2**, deployed as an interactive **Streamlit** web app.

Upload a photo of a skin condition and the app predicts whether it shows Acne or Eczema, along with a confidence score.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Retraining the Model](#retraining-the-model)
- [Requirements](#requirements)
- [How It Works](#how-it-works)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project builds a Convolutional Neural Network (CNN) to classify images of skin conditions into two categories:

- **Acne**
- **Eczema**

The model is built using **transfer learning**, starting from a MobileNetV2 backbone pre-trained on ImageNet, with a custom classification head trained on a labeled skin-condition image dataset. The final trained model (`best_model.keras`) is served through a lightweight Streamlit web application (`app.py`) that lets a user upload an image and get an instant prediction.

## Demo

The Streamlit app (`app.py`) provides a simple UI:

1. Upload a `.jpg`, `.jpeg`, or `.png` image of a skin condition.
2. Click **Predict**.
3. The app displays the predicted class (**Acne** or **Eczema**) and a confidence score as a percentage.

## Project Structure

```
CNN-image-classification-model-CO9-Lab10/
├── app.py               # Streamlit web app for inference
├── model.ipynb          # Jupyter notebook: EDA, preprocessing, training, evaluation
├── best_model.keras      # Trained/fine-tuned Keras model (MobileNetV2-based)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset

The notebook expects a dataset directory structured for `image_dataset_from_directory`, split into `train`, `val`, and `test` folders, each containing six subclasses:

```
Dataset/
├── train/
│   ├── Acne_Mild/
│   ├── Acne_Moderate/
│   ├── Acne_Severe/
│   ├── Eczema_Mild/
│   ├── Eczema_Moderate/
│   └── Eczema_Severe/
├── val/
│   └── (same subfolders)
└── test/
    └── (same subfolders)
```

The original dataset contains **6 fine-grained severity classes** (3 for Acne, 3 for Eczema). These are **remapped into a single binary label**:

- `Acne_Mild`, `Acne_Moderate`, `Acne_Severe` → **0 (Acne)**
- `Eczema_Mild`, `Eczema_Moderate`, `Eczema_Severe` → **1 (Eczema)**

Images are loaded at a batch size of 32 and resized to **224 × 224** pixels (the standard MobileNetV2 input size).

> **Note:** The dataset itself is not included in this repository. To retrain the model, you must supply your own `Dataset/train`, `Dataset/val`, and `Dataset/test` folders following the structure above, and update the `base_dir` path in `model.ipynb`.

## Model Architecture

The model uses **transfer learning** on top of **MobileNetV2**:

1. **Base model:** `MobileNetV2` pre-trained on ImageNet, loaded with `include_top=False` (the original 1000-class head is discarded), and initially **frozen** (`trainable = False`).
2. **Classification head:**
   - `GlobalAveragePooling2D` — condenses spatial feature maps into a single feature vector per image.
   - `Dropout(0.2)` — regularization to reduce overfitting.
   - `Dense(1, activation='sigmoid')` — outputs a single probability for binary classification.
3. **Loss/optimizer (initial phase):**
   - Optimizer: `Adam` with learning rate `1e-3`
   - Loss: `BinaryCrossentropy`
   - Metric: `accuracy`

### Data Augmentation

Applied only to the training set, via a `Sequential` augmentation pipeline:

- `RandomFlip("horizontal_and_vertical")`
- `RandomRotation(0.2)`
- `RandomZoom(0.1)`

All images (train/val/test) are rescaled to the `[0, 1]` range.

## Training Pipeline

Training happens in two phases, implemented in `model.ipynb`:

### Phase 1 — Feature Extraction (frozen base)
- The MobileNetV2 base is frozen; only the new classification head is trained.
- Up to 20 epochs, with:
  - `EarlyStopping` (monitors `val_loss`, patience 5, restores best weights)
  - `ReduceLROnPlateau` (monitors `val_loss`, factor 0.2, patience 3, min LR `1e-6`)

### Phase 2 — Fine-Tuning (unfrozen base)
- The entire MobileNetV2 base is unfrozen (`trainable = True`).
- The model is recompiled with a much smaller learning rate (`1e-5`) to avoid destroying the pre-trained weights.
- Training continues for additional epochs (fine-tuning phase), resuming from where phase 1 left off, using the same callbacks.

### Evaluation
After each phase, the notebook evaluates the model on the held-out test set and reports:
- Test loss and test accuracy
- A **confusion matrix**
- A full **classification report** (precision, recall, F1-score per class) via `scikit-learn`

The final fine-tuned model is saved as `best_model.keras`.

## Results

Exact metrics depend on the dataset used to train the model. The notebook (`model.ipynb`) contains:
- Loss/accuracy curves for both the initial training phase and the fine-tuning phase.
- A confusion matrix and classification report generated on the test split.

Re-run the notebook end-to-end on your dataset to reproduce these metrics for your own data.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/pj-hacks/CNN-image-classification-model-CO9-Lab10.git
   cd CNN-image-classification-model-CO9-Lab10
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

Make sure `best_model.keras` is present in the project root (it's already included in this repo), then run:

```bash
streamlit run app.py
```

This will open the app in your browser (typically at `http://localhost:8501`). From there:

1. Click **"Choose an image..."** and upload a `.jpg`, `.jpeg`, or `.png` file.
2. Click **Predict**.
3. View the predicted class (**Acne** or **Eczema**) and confidence score.

**Prediction logic (from `app.py`):**
- The uploaded image is converted to RGB, resized to 224×224, and normalized to `[0, 1]`.
- The model outputs a single sigmoid probability.
- If `probability > 0.5` → predicted class is **Eczema**, confidence = `probability * 100`.
- If `probability <= 0.5` → predicted class is **Acne**, confidence = `(1 - probability) * 100`.

### Retraining the Model

1. Open `model.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
2. Update the `base_dir` variable to point to your local `Dataset` folder.
3. Run all cells sequentially:
   - Data loading & binary label remapping
   - Exploratory Data Analysis (class distribution, sample visualization)
   - Data augmentation & preprocessing
   - Phase 1 training (frozen MobileNetV2 base)
   - Evaluation (confusion matrix, classification report)
   - Phase 2 fine-tuning (unfrozen base, low learning rate)
   - Final evaluation
   - Model export to `best_model.keras`
4. Replace the existing `best_model.keras` with your newly trained model to update the Streamlit app.

## Requirements

Listed in `requirements.txt`:

```
streamlit
tensorflow-cpu
Pillow
numpy
```

Additional packages used only in the training notebook (not required to run the app):
- `matplotlib`
- `scikit-learn`

Install them if you plan to run `model.ipynb`:

```bash
pip install matplotlib scikit-learn
```

## How It Works

```
User uploads image
        │
        ▼
Convert to RGB → Resize to 224x224 → Normalize [0,1] → Expand to batch of 1
        │
        ▼
MobileNetV2 (fine-tuned) + GlobalAveragePooling2D + Dropout + Dense(1, sigmoid)
        │
        ▼
Sigmoid probability (0.0 – 1.0)
        │
        ▼
   probability > 0.5 → "Eczema"
   probability ≤ 0.5 → "Acne"
        │
        ▼
Display prediction + confidence score
```

## Limitations & Disclaimer

- This model is a **binary classifier** trained to distinguish only between Acne and Eczema — it is **not** designed to detect any other skin condition and will force a prediction into one of these two categories even if the uploaded image shows something else entirely.
- This tool is built for **educational/demonstration purposes** as part of a lab exercise (CO9 Lab 10) and is **not a medical diagnostic tool**. It should not be used as a substitute for professional dermatological or medical advice.
- Model performance is directly tied to the quality, size, and diversity of the training dataset used, which is not included in this repository.

## Future Improvements

- Extend to multi-class classification (retain the original 6 severity sub-classes instead of collapsing to binary).
- Add more skin condition categories beyond Acne and Eczema.
- Introduce explainability (e.g., Grad-CAM) to visualize which regions of the image drove the prediction.
- Add automated tests and CI for the Streamlit app.
- Containerize the app with Docker for easier deployment.

---
  
Participants:

- Joseph Prince Aniekeme 22/EG/cO/1774
- Edem, Etimbuk Akaninyene 22/EG/CO/1694
- Eguaikhe, Esther Ikpemiosime 22/EG/CO/1634
- Emah, Victor Victor 22/Eg/Co/1654
- Emah, Etido Udofia 22/EG/CO/1674
- Ekpenyong, Joshua Effiong 22/EG/CO/176
- Iboroma Michael Daniel 22/EG/CO/1754
- Akpan Goodness Bright 22/EG/CO/1704
- Effiong, Wisdom Paulinus 22/EG/CO/1794
- Okon Abasiofiok Monday  22/EG/CO/1804
---


## License

No license file is currently included in this repository. If you intend to reuse this code, please check with the repository owner or add an appropriate license.

