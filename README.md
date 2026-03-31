# X-Ray Pneumonia Detector (XPD)

Deep learning system for automated pneumonia detection from chest X-ray images. Uses transfer learning with multiple CNN architectures, Grad-CAM explainability, and comprehensive evaluation metrics.

## Highlights

- **Multi-architecture comparison**: ResNet34, ResNet50, DenseNet121
- **Grad-CAM explainability**: Visual explanations showing which X-ray regions drive predictions
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, PR curves
- **Interactive demo**: Streamlit app with real-time Grad-CAM overlay and confidence thresholding
- **Proper ML pipeline**: LR finder, 2-stage fine-tuning, held-out test evaluation

---

## Results

### Model Performance Comparison

![Metrics Table](outputs/figures/metrics_table.png)

**DenseNet121** achieves the best overall performance with **94.23% accuracy** and **0.9812 AUC-ROC**, while using only **7.0M parameters** — 3x fewer than ResNet50.

![Metrics Bar Chart](outputs/figures/metrics_bar_chart.png)

### Per-Class Classification Report

![Per-Class Metrics](outputs/figures/per_class_metrics.png)

### Confusion Matrices

![Confusion Matrices](outputs/figures/confusion_matrices.png)

### ROC Curves

![ROC Curves](outputs/figures/roc_curves.png)

### Precision-Recall Curves

![PR Curves](outputs/figures/pr_curves.png)

### Training Curves

Loss and accuracy progression during 2-stage fine-tuning (dotted line marks backbone unfreeze point):

![Training Curves](outputs/figures/training_curves.png)

---

## Architecture

```
                                  +-------------------+
                                  |  Chest X-Ray      |
                                  |  Image Input      |
                                  +---------+---------+
                                            |
                                  +---------v---------+
                                  |  Preprocessing     |
                                  |  Resize + Augment  |
                                  |  + Normalize       |
                                  +---------+---------+
                                            |
                           +----------------+----------------+
                           |                |                |
                    +------v------+  +------v------+  +------v------+
                    |  ResNet34   |  |  ResNet50   |  | DenseNet121 |
                    |  (21.3M)    |  |  (23.5M)    |  |  (7.0M)     |
                    +------+------+  +------+------+  +------+------+
                           |                |                |
                           +----------------+----------------+
                                            |
                                  +---------v---------+
                                  |  FC Classifier     |
                                  |  Head              |
                                  +---------+---------+
                                            |
                              +-------------+-------------+
                              |                           |
                     +--------v--------+        +---------v-------+
                     |   Prediction    |        |   Grad-CAM      |
                     |  NORMAL /       |        |   Heatmap       |
                     |  PNEUMONIA      |        |   Overlay       |
                     +-----------------+        +-----------------+
```

---

## Project Structure

```
xray-classifier/
├── config/
│   └── config.yaml              # Hyperparameters and paths
├── src/
│   ├── data.py                  # Data loading, splits, augmentation
│   ├── model.py                 # Architecture factory
│   ├── train.py                 # Training pipeline (LR finder, 2-stage fine-tuning)
│   ├── evaluate.py              # Metrics, plots, architecture comparison
│   └── gradcam.py               # Grad-CAM from scratch (PyTorch hooks)
├── app/
│   └── app.py                   # Streamlit web app with Grad-CAM
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_training.ipynb        # Interactive training
│   └── 03_evaluation.ipynb      # Evaluation with visualizations
├── scripts/
│   └── generate_figures.py      # Generate evaluation plots
├── outputs/
│   ├── models/                  # Saved model checkpoints
│   ├── figures/                 # Generated plots
│   └── metrics/                 # JSON metric files
├── tests/
│   └── test_model.py            # Unit tests (7 tests)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle and extract it:

```bash
# Using Kaggle CLI
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/chest_xray
```

Expected structure:
```
data/chest_xray/
├── train/
│   ├── NORMAL/       (1,341 images)
│   └── PNEUMONIA/    (3,875 images)
├── val/
└── test/
    ├── NORMAL/       (234 images)
    └── PNEUMONIA/    (390 images)
```

### 3. Train Models

**Option A: Using the training script**
```bash
python -m src.train
```

**Option B: Using the notebook (recommended for exploration)**
```bash
jupyter notebook notebooks/02_training.ipynb
```

### 4. Evaluate

```bash
python -m src.evaluate
```

This generates:
- Confusion matrices, ROC curves, PR curves in `outputs/figures/`
- Metric JSON files in `outputs/metrics/`
- Architecture comparison table in the terminal

### 5. Run the App

```bash
streamlit run app/app.py
```

---

## Training Approach

### Data Pipeline
- **Dataset**: 5,216 training images (1,341 NORMAL, 3,875 PNEUMONIA)
- **Split**: 80/20 random split of training data; test set (624 images) held out entirely
- **Augmentation**: Random rotation (15 deg), horizontal flip, lighting adjustment, warp, with ImageNet normalization
- **Note**: `flip_vert=False` because vertical flips are not anatomically meaningful for chest X-rays

### Transfer Learning Strategy
1. **LR Finder**: Automatically determine optimal learning rate (valley method)
2. **Stage 1**: Train classifier head with frozen backbone (3 epochs)
3. **Stage 2**: Unfreeze all layers, train with discriminative learning rates — lower layers get 100x smaller LR (5 epochs)

### Grad-CAM Implementation
Implemented from scratch using PyTorch hooks (~40 lines of core logic):
- Forward hook captures activations at the last convolutional layer
- Backward hook captures gradients
- Heatmap = ReLU(weighted sum of activations by global-average-pooled gradients)
- No external Grad-CAM library dependency

---

## Limitations & Future Work

### Current Limitations
- **Single-center dataset**: Model trained on data from a single institution — may not generalize to X-rays from different machines, patient demographics, or imaging protocols
- **Binary classification only**: Does not distinguish between bacterial and viral pneumonia
- **Class imbalance**: ~3:1 pneumonia-to-normal ratio in training data
- **No clinical validation**: Not validated against radiologist diagnoses in a clinical setting

### Potential Improvements
- Train on multi-center datasets (CheXpert, MIMIC-CXR) for better generalization
- Add multi-class classification (Normal / Bacterial / Viral pneumonia)
- Implement uncertainty quantification (MC Dropout, ensemble methods)
- Add fairness analysis across demographic groups
- Integrate DICOM metadata for richer input features
- Experiment with Vision Transformers (ViT, DeiT)

---

## Disclaimer

This tool is for **educational and research purposes only**. It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

## License

MIT
