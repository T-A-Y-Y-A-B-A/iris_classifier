# ============================================================
# PROJECT 2: Data Classification Using AI
# STEP 2: Feature Scaling + Train-Test Split
# ============================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
iris = load_iris()
X = iris.data    # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=setosa, 1=versicolor, 2=virginica

print("=" * 55)
print("   STEP 2: SCALING + TRAIN-TEST SPLIT")
print("=" * 55)

# --- Step A: Split BEFORE scaling (prevent data leakage) ---
# 80% training, 20% testing | shuffle=True removes order bias
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\n📊 Total samples   : {len(X)}")
print(f"🎓 Training set    : {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"🧪 Testing set     : {len(X_test)} samples  ({len(X_test)/len(X)*100:.0f}%)")

# --- Step B: Feature Scaling (StandardScaler) ---
# Fit ONLY on training data, then transform both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # ← use transform only, not fit_transform

print("\n--- Before Scaling (X_train sample) ---")
print(X_train[:3].round(2))

print("\n--- After Scaling (X_train_scaled sample) ---")
print(X_train_scaled[:3].round(2))

print(f"\n📐 Scaler Mean     : {scaler.mean_.round(2)}")
print(f"📐 Scaler Variance : {scaler.var_.round(2)}")

print("\n✅ Step 2 Complete: Data split and scaled.")
print("   Saving scaler for reuse in later steps...")

# Save for reuse
import joblib, os
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
np.save("models/X_train_scaled.npy", X_train_scaled)
np.save("models/X_test_scaled.npy",  X_test_scaled)
np.save("models/y_train.npy", y_train)
np.save("models/y_test.npy",  y_test)
print("   Saved: models/scaler.pkl, X_train_scaled.npy, X_test_scaled.npy")
