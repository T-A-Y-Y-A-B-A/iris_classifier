# ============================================================
# PROJECT 2: Data Classification Using AI
# STEP 5: run_all.py — Full Pipeline (One Command)
# ============================================================
# Run this file to execute all steps end to end:
#   python run_all.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score
)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 ─ LOAD & EXPLORE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 1 — LOAD & EXPLORE DATASET")
print("="*55)

iris = load_iris()
X, y = iris.data, iris.target
class_names = list(iris.target_names)

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [class_names[i] for i in y]

print(f"Samples   : {len(X)}")
print(f"Features  : {X.shape[1]}  → {iris.feature_names}")
print(f"Classes   : {class_names}")
print(f"\nClass balance:\n{df['species'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 ─ SPLIT + SCALE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 2 — TRAIN-TEST SPLIT + FEATURE SCALING")
print("="*55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Training samples : {len(X_train)}  (80%)")
print(f"Testing  samples : {len(X_test)}   (20%)")
print(f"Scaler Mean      : {scaler.mean_.round(2)}")

# ─────────────────────────────────────────────────────────────
# STEP 3 ─ FIND OPTIMAL K + TRAIN
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 3 — KNN: FIND BEST K + TRAIN")
print("="*55)

k_values, train_accs, test_accs = [], [], []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    k_values.append(k)
    train_accs.append(knn.score(X_train_s, y_train))
    test_accs.append(knn.score(X_test_s, y_test))

best_k = k_values[test_accs.index(max(test_accs))]
print(f"Optimal K = {best_k}  |  Test Accuracy = {max(test_accs):.4f}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_s, y_train)
joblib.dump(model,  "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ─────────────────────────────────────────────────────────────
# STEP 4 ─ EVALUATE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  STEP 4 — EVALUATE")
print("="*55)

preds    = model.predict(X_test_s)
accuracy = accuracy_score(y_test, preds)
f1_macro = f1_score(y_test, preds, average='macro')
cm       = confusion_matrix(y_test, preds)
f1_each  = f1_score(y_test, preds, average=None)

print(f"Accuracy         : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"F1 Score (macro) : {f1_macro:.4f}")
print("\n" + classification_report(y_test, preds, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# STEP 5 ─ VISUALIZE (4-panel report)
# ─────────────────────────────────────────────────────────────
print("="*55)
print("  STEP 5 — GENERATING CHARTS")
print("="*55)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Project 2 — KNN Iris Classifier: Full Report", fontsize=15, fontweight='bold')

# Panel 1: K vs Accuracy (Elbow)
axes[0, 0].plot(k_values, train_accs, 'o-', label='Train Acc', color='steelblue')
axes[0, 0].plot(k_values, test_accs,  's-', label='Test Acc',  color='darkorange')
axes[0, 0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0, 0].set_title("Elbow Method: Choosing K")
axes[0, 0].set_xlabel("K Value")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1], linewidths=0.5)
axes[0, 1].set_title("Confusion Matrix")
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Actual")

# Panel 3: F1 Score per class
colors = ['#2ecc71' if v >= 0.95 else '#e67e22' if v >= 0.85 else '#e74c3c' for v in f1_each]
bars = axes[1, 0].bar(class_names, f1_each, color=colors, edgecolor='black', width=0.5)
axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].set_title(f"F1 Score per Class  (macro avg = {f1_macro:.2f})")
axes[1, 0].set_ylabel("F1 Score")
axes[1, 0].axhline(1.0, linestyle='--', color='gray', alpha=0.5)
for bar, val in zip(bars, f1_each):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha='center', fontweight='bold')

# Panel 4: Scatter (Petal features, test set — colored by prediction)
colors_map = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}
for cls in range(3):
    mask = preds == cls
    axes[1, 1].scatter(
        X_test[mask, 2], X_test[mask, 3],
        c=colors_map[cls], label=class_names[cls],
        edgecolors='black', linewidth=0.5, s=70, alpha=0.85
    )
# Mark wrong predictions
wrong = preds != y_test
axes[1, 1].scatter(X_test[wrong, 2], X_test[wrong, 3],
                   s=200, facecolors='none', edgecolors='black',
                   linewidths=2, label='Misclassified')
axes[1, 1].set_title("Predictions: Petal Length vs Width")
axes[1, 1].set_xlabel("Petal Length (cm)")
axes[1, 1].set_ylabel("Petal Width (cm)")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("models/full_report.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Full report chart saved → models/full_report.png")
print("🤖 Model saved            → models/knn_model.pkl")
print("📐 Scaler saved           → models/scaler.pkl")
print("\n🎉 PROJECT 2 COMPLETE!\n")
