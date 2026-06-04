# ============================================================
# PROJECT 2: Data Classification Using AI
# STEP 4: Evaluate — Confusion Matrix + F1 Score
# ============================================================

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)

# --- Load model and data ---
model   = joblib.load("models/knn_model.pkl")
X_test  = np.load("models/X_test_scaled.npy")
y_test  = np.load("models/y_test.npy")

class_names = ['setosa', 'versicolor', 'virginica']
predictions = model.predict(X_test)

print("=" * 55)
print("   STEP 4: MODEL EVALUATION")
print("=" * 55)

# --- Core Metrics ---
accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average='macro')
f1_weighted = f1_score(y_test, predictions, average='weighted')

print(f"\n✅ Accuracy         : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"✅ F1 Score (macro) : {f1_macro:.4f}")
print(f"✅ F1 Score (weighted): {f1_weighted:.4f}")

print("\n--- Full Classification Report ---")
print(classification_report(y_test, predictions, target_names=class_names))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, predictions)
print("--- Confusion Matrix (raw) ---")
print(cm)

# --- Plot Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Project 2: KNN Iris Classifier — Evaluation", fontsize=14, fontweight='bold')

# Heatmap
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    ax=axes[0], linewidths=0.5
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# F1 Score per class bar chart
f1_per_class = f1_score(y_test, predictions, average=None)
colors = ['#2ecc71' if v >= 0.95 else '#e67e22' if v >= 0.85 else '#e74c3c' for v in f1_per_class]
bars = axes[1].bar(class_names, f1_per_class, color=colors, edgecolor='black', width=0.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("F1 Score per Class")
axes[1].set_ylabel("F1 Score")
for bar, val in zip(bars, f1_per_class):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("models/evaluation_report.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Chart saved to models/evaluation_report.png")
print("✅ Step 4 Complete: Evaluation done.")
