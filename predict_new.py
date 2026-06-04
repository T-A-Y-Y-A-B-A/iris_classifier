# ============================================================
# PROJECT 2: Data Classification Using AI
# BONUS: predict_new.py — Predict on Your Own Data
# ============================================================
# Usage:
#   python predict_new.py
# Edit `new_samples` below to test your own flower measurements.

import numpy as np
import joblib

# --- Load saved model and scaler ---
model  = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
class_names = ['setosa', 'versicolor', 'virginica']

# ─────────────────────────────────────────────────────────────
# 🌸 ADD YOUR OWN SAMPLES HERE
# Format: [sepal_length, sepal_width, petal_length, petal_width]  (all in cm)
# ─────────────────────────────────────────────────────────────
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Classic setosa
    [6.7, 3.0, 5.2, 2.3],   # Likely virginica
    [5.8, 2.7, 4.1, 1.0],   # Likely versicolor
    [4.9, 3.1, 1.5, 0.1],   # Setosa
    [7.0, 3.2, 6.0, 1.8],   # Virginica
])

# --- Scale using the SAME scaler fitted on training data ---
new_samples_scaled = scaler.transform(new_samples)

# --- Predict ---
predictions  = model.predict(new_samples_scaled)
probabilities = model.predict_proba(new_samples_scaled)   # requires sklearn >= 0.9

print("=" * 65)
print("  PREDICTION RESULTS")
print("=" * 65)
print(f"{'#':>3} | {'SL':>5} {'SW':>5} {'PL':>5} {'PW':>5} | {'Prediction':>12} | Confidence")
print("-" * 65)

for i, (sample, pred) in enumerate(zip(new_samples, predictions)):
    sl, sw, pl, pw = sample
    proba = probabilities[i]
    confidence = proba[pred]
    print(f"{i+1:>3} | {sl:>5.1f} {sw:>5.1f} {pl:>5.1f} {pw:>5.1f} | {class_names[pred]:>12} | {confidence*100:.1f}%")

print("\n✅ Done. Edit `new_samples` in this file to test your own values.")
