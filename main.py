# =========================
# Import Modules
# =========================
import sys
import os

sys.path.append(os.path.abspath("./src/Medical"))
from data_loader import load_data
from preprocessing import preprocess_data
from windowing import create_sequences
from model import build_lstm_autoencoder

import numpy as np
import pandas as pd

# =========================
# 1. Load raw IoT data
# =========================

# Define your local data path (change this to your own path)
folder_path = "data/medical/set-a/"
# Load all patient data
df_all = load_data(folder_path)
# Print shape and sample data to verify
print("Raw data shape:", df_all.shape)
print(df_all.head())

# =========================
# 2. Preprocessing pipeline
# =========================

# Apply the full preprocessing pipeline to raw IoT ICU data
# This step transforms raw sensor readings into a clean, structured,
# and machine-learning-ready time-series dataset.
# It includes:
# - Data pivoting (long → wide format)
# - Missing value handling
# - Feature scaling (patient-wise normalization)
# - Missingness indicator integration
df_final = preprocess_data(df_all)


# Display the shape of the processed dataset
print("Processed data shape:", df_final.shape)
# Preview the first rows of the cleaned dataset
print(df_final.head())



# =========================================================
# 3. Windowing (sequence generation)
# =========================================================

# Select features (exclude missing flags if needed)
feature_cols = [c for c in df_final.columns if not c.endswith("_missing")]

# Generate sequences
X, pids = create_sequences(df_final, feature_cols, window_size=6)

print("Final X shape (model input):", X.shape)


# =========================================================
# Assumption
# =========================================================
# At this stage, we already have:
# X     → input sequences (shape: num_samples, timesteps, features)
# pids  → patient_id corresponding to each sequence

# NOTE:
# These are produced from preprocessing + windowing steps

# =========================================================
# 1. Extract Input Shape
# =========================================================

# Number of time steps per sequence (window size)
timesteps = X.shape[1]

# Number of features per time step
features = X.shape[2]

# =========================================================
# 2. Build the Model
# =========================================================

# Build LSTM Autoencoder model
# The model learns normal physiological patterns
model = build_lstm_autoencoder(timesteps, features)

# Print architecture (important for report/debugging)
model.summary()

# =========================================================
# 3. Train the Model
# =========================================================

# Train the model to reconstruct normal sequences
# Input = Output because this is an autoencoder
history = model.fit(
    X, X,                  # autoencoder learns reconstruction
    epochs=10,             # number of training iterations
    batch_size=32,         # batch size
)
# After training:
# The model becomes good at reconstructing "normal patterns"


# =========================================================
# 4. Reconstruction Phase
# =========================================================

# The model tries to reconstruct the same sequences
# This simulates how it behaves on incoming real-time data
X_pred = model.predict(X)


# =========================================================
# 5. Compute Reconstruction Error (Loss)
# =========================================================

# Calculate Mean Squared Error between original and reconstructed sequences
# axis=(1,2) → compute ONE value per sequence
loss = np.mean((X - X_pred) ** 2, axis=(1, 2))

# Interpretation:
# Low loss  → normal pattern
# High loss → abnormal (anomaly)

# =========================================================
# Create results container
# =========================================================

results = pd.DataFrame({
    "patient_id": pids,
    "loss": loss
})

# =========================================================
# 6. Patient-specific threshold
# =========================================================

# Purpose:
# Instead of using a global threshold for all patients,
# we compute a separate threshold for each patient.
# This makes the system more realistic for healthcare applications
# because each patient has a unique physiological baseline.


# =========================================================
# Compute patient-specific thresholds
# =========================================================

# For each patient, we compute the 95th percentile of reconstruction loss
# This represents the upper bound of "normal behavior" for that patient

patient_thresholds = results.groupby("patient_id")["loss"].quantile(0.95)

# =========================================================
# Map thresholds back to each sequence
# =========================================================

# Each window belongs to a specific patient
# We assign the corresponding patient threshold to each row

results["threshold"] = results["patient_id"].map(patient_thresholds)

# =========================================================
# 7. Detect anomalies using patient baseline
# =========================================================

# A sequence is considered anomalous if its loss exceeds
# the patient's own threshold (not a global one)

results["anomaly"] = results["loss"] > results["threshold"]

# =========================================================
# Compute Risk Score
# =========================================================

# Risk score represents how far the current behavior
# deviates from the patient's normal baseline

results["risk_score"] = results["loss"] / results["threshold"]

# =========================================================
# 8. REAL-TIME ALERT GENERATION MODULE (WINDOW LEVEL)
# =========================================================

# In ICU systems, patient data arrives continuously from IoT devices.
# Each time window is processed independently to detect abnormalities immediately.
# This module converts risk scores into clinically interpretable alerts.


# =========================================================
# Alert Decision Function
# =========================================================

def generate_alert(risk_score):
    """
    Convert numerical risk score into a clinically interpretable alert.

    This simulates ICU decision support systems where ML outputs
    are translated into actionable clinical states.
    """

    # ---------------------------------------------------------
    # CRITICAL CASE
    # ---------------------------------------------------------
    # Strong deviation from patient's baseline behavior.
    # In real ICU:
    # - immediate clinical attention is required
    # - patient condition may be unstable
    if risk_score > 1.2:
        return "CRITICAL"

    # ---------------------------------------------------------
    # WARNING CASE
    # ---------------------------------------------------------
    # Moderate deviation from normal behavior.
    # In real ICU:
    # - early sign of deterioration
    # - requires close monitoring
    elif risk_score > 1.0:
        return "WARNING"

    # ---------------------------------------------------------
    # NORMAL CASE
    # ---------------------------------------------------------
    # Patient behavior is within expected range.
    # No immediate action required.
    else:
        return "NORMAL"


# =========================================================
# Apply alert function to each window (real-time simulation)
# =========================================================

# Each row in results represents:
# one patient window + its computed risk score

results["alert"] = results["risk_score"].apply(generate_alert)


# =========================================================
# 9. PATIENT-LEVEL CLINICAL SUMMARY (LONG-TERM VIEW)
# =========================================================

# In a real ICU environment:
# - Doctors do NOT analyze every single time window manually
# - Instead, they need a summarized view of each patient's condition over time

# At this stage, we move from:
# WINDOW-LEVEL (instant detection)
# TO PATIENT-LEVEL (clinical assessment)

# ---------------------------------------------------------
# What we are doing here:
# ---------------------------------------------------------
# We aggregate all window-level risk scores for each patient
# to estimate their overall health condition.

patient_risk = results.groupby("patient_id")["risk_score"].mean()

# =========================================================
# PATIENT PRIORITIZATION (ICU TRIAGE SYSTEM)
# =========================================================

# After computing overall risk per patient,
# we sort patients to identify who needs attention first

# This simulates ICU triage:
# patients with higher risk are prioritized

priority_ranking = patient_risk.sort_values(ascending=False)



# =========================================================
# 10. FINAL CLINICAL OUTPUT (ICU MONITORING VIEW)
# =========================================================

# This section represents the final layer of the system.
# In real-world ICU systems, this is what clinicians interact with.

# Instead of raw model outputs,
# we present structured, interpretable clinical information.


# =========================================================
# (1) ICU PRIORITY LIST
# =========================================================

# We sort patients based on their overall risk score
# to identify which patients require immediate medical attention

# In real ICU:
# - Nurses/doctors check this list first
# - Highest risk patients are prioritized for intervention

print("=== ICU PRIORITY LIST (High Risk Patients) ===")

# Show top 10 highest-risk patients
print(priority_ranking.head(10))


# =========================================================
# (2) PATIENT RISK SUMMARY STATISTICS
# =========================================================

# This provides a statistical overview of patient risk distribution

# In real clinical systems:
# - Helps understand overall ICU load
# - Identifies whether most patients are stable or critical

print("\n=== Patient Risk Summary Statistics ===")

# Displays:
# - mean risk
# - std deviation
# - min / max risk
print(patient_risk.describe())



# =========================================================
# (3) SAMPLE REAL-TIME ALERTS (WINDOW LEVEL)
# =========================================================

# This simulates real-time monitoring output
# where each time window generates a clinical decision

# In real ICU systems:
# - Each patient monitor continuously sends alerts
# - Doctors receive "WARNING / CRITICAL / NORMAL" signals

print("\n=== Sample Window-Level Alerts ===")

# Show sample of:
# - patient_id
# - risk_score (severity level)
# - alert (clinical decision)

print(results[["patient_id", "risk_score", "alert"]].head(10))

# =========================================================
# 11. SAVE RESULTS (FOR REPORT / DASHBOARD / DEPLOYMENT)
# =========================================================

# In real systems:
# - results are stored in database or logging system
# - doctors may revisit historical patient states

# Here we save results for later analysis

results.to_csv("outputs/window_level_results.csv", index=False)


# Save patient-level risk ranking

patient_risk.to_csv("outputs/patient_risk_scores.csv")


# =========================================================
# 12. BASIC SYSTEM INSIGHT (EVALUATION)
# =========================================================

print("\nTotal windows:", len(results))
print("Detected anomalies:", results["anomaly"].sum())
print("Anomaly percentage:", results["anomaly"].mean() * 100)