# =========================================================
# SECURITY INFERENCE MODULE
# =========================================================
# This file is responsible for deploying the trained
# XGBoost model to detect cyber attacks in IoT sensor data.
#
# In real-world systems:
# - IoT devices send network/data packets continuously
# - This module checks if the data is malicious or normal
# - If attack is detected → security alert is triggered
# - If normal → data is passed to medical system
# =========================================================

# ==============================
# Import Libraries
# ==============================
import numpy as np
import joblib
# joblib → used to load the trained ML model from disk

from sklearn.preprocessing import LabelEncoder
# used to decode predicted numeric labels back to attack names

# =========================================================
# MAIN FUNCTION: ATTACK DETECTION
# =========================================================

def detect_attacks(X_new):
    """
    This function takes NEW incoming IoT data
    and returns predicted attack labels.

    PARAMETERS:
    ----------
    X_new : numpy array
        New sensor/network data coming from IoT devices

    RETURNS:
    -------
    attack_labels : array of strings
        Predicted label for each sample
        (e.g., 'BenignTraffic', 'DDoS', 'MITM', etc.)
    """
    # =========================================================
    # 1. LOAD TRAINED MODEL
    # =========================================================
    # We load the XGBoost model that was trained earlier
    # This model already learned patterns of attacks vs normal traffic
    model = joblib.load("../../models/xgboost_model.pkl")

    # =========================================================
    # 2. LOAD LABEL ENCODER
    # =========================================================
    # We load the same encoder used during training
    # so we can correctly convert numeric predictions back to labels
    le = joblib.load("../../models/label_encoder.pkl")

    # =========================================================
    # 4. MODEL PREDICTION
    # =========================================================
    # We pass the new data into the trained model
    # Model outputs a numeric class for each sample
    y_pred = model.predict(X_new)

    # =========================================================
    # 5. CONVERT PREDICTIONS TO READABLE LABELS
    # =========================================================

    # Convert numeric predictions into attack names
    attack_labels = le.inverse_transform(y_pred)
    # =====================================================
    # 6. RETURN RESULTS
    # =====================================================
    return attack_labels

