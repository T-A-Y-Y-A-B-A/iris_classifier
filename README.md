# Project 2: Data Classification Using AI
**DecodeLabs Industrial Training Kit · Batch 2026**

> Build a basic classification model using a small dataset — mastering the fundamental supervised learning pipeline.

---

## Overview

This project implements a **K-Nearest Neighbors (KNN)** classifier on the classic **Iris dataset**. You will load and explore data, apply feature scaling, split into train/test sets, train the model, and evaluate it using a confusion matrix and F1 score.

**Algorithm:** K-Nearest Neighbors (KNN)  
**Dataset:** Iris (150 samples · 3 classes · 4 features)  
**Framework:** scikit-learn

---

## Project Structure

```
project2/
├── step1_load_data.py       # Load & explore the Iris dataset
├── step2_split_scale.py     # Train-test split + StandardScaler
├── step3_train_model.py     # Elbow method + KNN training
├── step4_evaluate.py        # Confusion matrix + F1 score + charts
├── run_all.py               # Run the full pipeline in one command
├── predict_new.py           # Predict on your own custom inputs
├── project2_iris_classifier.html  # Interactive browser dashboard
└── README.md                # This file
```

---

## IPO Framework

| Stage | Details |
|-------|---------|
| **Input** | Iris dataset, feature scaling via `StandardScaler` |
| **Process** | Train-test split (80/20), KNN algorithm, elbow method for K |
| **Output** | Confusion matrix, F1 score, predictions |

---

## Setup

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## How to Run

### Option 1 — Full pipeline (recommended)
```bash
python run_all.py
```
Runs all 5 steps end-to-end and saves a `full_report.png` chart.

### Option 2 — Step by step (best for learning)
```bash
python step1_load_data.py      # Explore the dataset
python step2_split_scale.py    # Split and scale the data
python step3_train_model.py    # Find best K and train
python step4_evaluate.py       # Evaluate and visualize
```

### Option 3 — Predict on your own data
Edit the `new_samples` array in `predict_new.py`:
```python
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # [sepal_length, sepal_width, petal_length, petal_width]
])
```
Then run:
```bash
python predict_new.py
```

### Option 4 — Interactive HTML dashboard
Open `project2_iris_classifier.html` in any browser — no server required.

---

## The Dataset

The Iris dataset contains measurements of 150 iris flowers across 3 species:

| Feature | Description |
|---------|-------------|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |

**Classes:** Setosa (0) · Versicolor (1) · Virginica (2) — 50 samples each (balanced).

---

## Key Concepts

### Why scale features?
KNN uses distance to find neighbors. Without scaling, a feature ranging 0–1000 dominates one ranging 0–1, creating bias. `StandardScaler` transforms all features to mean=0, variance=1.

> **Rule:** Always fit the scaler on training data only, then transform both sets. Fitting on test data causes data leakage.

### Why the Elbow Method?
- Small K (e.g. K=1) → overfitting, sensitive to noise
- Large K (e.g. K=100) → underfitting, too generic
- The elbow point is where test accuracy stops improving significantly

### Why F1 Score over Accuracy?
In imbalanced datasets, a model can achieve 99% accuracy by predicting only the majority class. F1 score balances **precision** (no false alarms) and **recall** (no missed detections), giving a truer picture of performance.

---

## Results

```
Optimal K   : 1
Accuracy    : 100.0%
F1 (macro)  : 1.0000

              precision  recall  f1-score  support
    setosa       1.00     1.00     1.00       10
versicolor       1.00     1.00     1.00        9
 virginica       1.00     1.00     1.00       11
```

The Iris dataset is linearly separable (especially Setosa), which is why KNN achieves perfect accuracy on it — making it an ideal dataset for learning the supervised learning pipeline.

---

## Outputs Generated

| File | Description |
|------|-------------|
| `models/knn_model.pkl` | Trained KNN model |
| `models/scaler.pkl` | Fitted StandardScaler |
| `models/full_report.png` | 4-panel evaluation chart |

---

## Skills Demonstrated

- Data handling with `pandas` and `numpy`
- Supervised learning pipeline with `scikit-learn`
- Feature engineering (StandardScaler)
- Hyperparameter tuning (Elbow method for K)
- Model evaluation (confusion matrix, F1 score, classification report)
- Model persistence with `joblib`

---

## Next Steps

Once you complete this project, explore:
- Comparing KNN against Decision Tree and SVM on the same dataset
- Testing on a dataset with class imbalance to see why F1 matters
- Moving to **Project 3** — Deep Learning & CNNs (image classification)

---

*Powered by DecodeLabs · decodelabs.tech@gmail.com · www.decodelabs.tech*
