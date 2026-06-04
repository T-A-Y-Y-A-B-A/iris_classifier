# ============================================================
# PROJECT 2: Data Classification Using AI
# STEP 3: Train the KNN Classifier
# ============================================================

import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

# --- Load preprocessed data from Step 2 ---
X_train = np.load("models/X_train_scaled.npy")
X_test  = np.load("models/X_test_scaled.npy")
y_train = np.load("models/y_train.npy")
y_test  = np.load("models/y_test.npy")

class_names = ['setosa', 'versicolor', 'virginica']

print("=" * 55)
print("   STEP 3: TRAIN KNN CLASSIFIER")
print("=" * 55)

# --- Find Optimal K using Elbow Method ---
print("\n🔍 Finding Optimal K (Elbow Method)...")
print(f"{'K':>4} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 30)

best_k, best_acc = 1, 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc = knn.score(X_train, y_train)
    test_acc  = knn.score(X_test,  y_test)
    marker = " ← best" if test_acc > best_acc else ""
    if test_acc > best_acc:
        best_acc, best_k = test_acc, k
    print(f"{k:>4} | {train_acc:>10.4f} | {test_acc:>10.4f}{marker}")

print(f"\n🏆 Optimal K = {best_k}  (Test Accuracy = {best_acc:.4f})")

# --- Train Final Model with Best K ---
print(f"\n🚀 Training final KNN model with K={best_k}...")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

# --- Quick Sanity Check ---
predictions = model.predict(X_test)
print("\n--- Sample Predictions vs Actual ---")
print(f"{'#':>3} | {'Predicted':>12} | {'Actual':>12} | {'Match':>5}")
print("-" * 40)
for i in range(10):
    pred   = class_names[predictions[i]]
    actual = class_names[y_test[i]]
    match  = "✅" if predictions[i] == y_test[i] else "❌"
    print(f"{i+1:>3} | {pred:>12} | {actual:>12} | {match:>5}")

# --- Save the trained model ---
joblib.dump(model, "models/knn_model.pkl")
print(f"\n✅ Step 3 Complete: Model trained and saved to models/knn_model.pkl")
