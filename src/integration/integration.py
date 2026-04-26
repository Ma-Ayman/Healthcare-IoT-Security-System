# =========================================================
# INTEGRATION MODULE
# =========================================================
# This file connects:
# 1. Security Model (XGBoost - Attack Detection)
# 2. Medical Model (LSTM Autoencoder - Risk & Anomaly)
#
# It simulates a REAL IoT Healthcare System pipeline:
# - Data comes from sensors
# - First → security check
# - Then → medical analysis
# =========================================================

# ==============================
# Import Libraries
# ==============================

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("../Security"))
# Import security detection function
from security_inference import detect_attacks

# =========================================================
# 1. LOAD SIMULATED INPUT DATA
# =========================================================
# In real systems:
# data comes from IoT devices (real-time streaming)
#
# Here:
# we simulate this using saved dataset

X_new = np.load("../../data/security/security_dataset.npz")["X"][:100]
# take only 100 samples to simulate streaming

# =========================================================
# 2. RUN SECURITY CHECK
# =========================================================
# Step 1 in pipeline:
# Check if incoming data is malicious or safe

attack_labels = detect_attacks(X_new)



# =========================================================
# 3. DECISION LOGIC (SECURITY LAYER)
# =========================================================
# We now decide what to do with each data sample

secure_data_indices = []   # indices of SAFE data
security_alerts = []       # list of detected attacks
security_log = []   #full log for dashboard

for i, label in enumerate(attack_labels):

    # -----------------------------------------------------
    # CASE 1: ATTACK DETECTED
    # -----------------------------------------------------
    if label != "BenignTraffic":

        # store alert
        security_alerts.append({
            "sample_id": i,
            "attack_type": label,
            "status": "ATTACK"
        })
        # add to full log
        security_log.append({
            "sample_id": i,
            "traffic_type": "ATTACK",
            "attack_type": label,
            "status": "COMPROMISED"
        })

        print(f"🚨 SECURITY ALERT → Sample {i} | Attack: {label}")
    # -----------------------------------------------------
    # CASE 2: SAFE DATA
    # -----------------------------------------------------
    else:
        # keep safe data for medical analysis
        secure_data_indices.append(i)
        # 👇 add to full log
        security_log.append({
            "sample_id": i,
            "traffic_type": "NORMAL",
            "attack_type": "BenignTraffic",
            "status": "SAFE"
        })
# =========================================================
# 4. FILTER SAFE DATA ONLY
# =========================================================
# Only clean (non-attacked) data goes to medical system

X_safe = X_new[secure_data_indices]

print("\nTotal Safe Samples:", len(X_safe))
print("Total Attacks Detected:", len(security_alerts))


# =========================================================
# 5. DECISION: ALLOW OR BLOCK MEDICAL SYSTEM
# =========================================================
# IMPORTANT CONCEPT:
# We DO NOT pass security data to medical system
#
# Instead:
# - Security module validates device integrity
# - If device is SAFE → allow medical system to run
# - If device is ATTACKED → block medical analysis
#
# This simulates real-world trusted data pipelines


# =========================================================
# CASE 1: DEVICE IS SAFE (NO ATTACK DETECTED)
# =========================================================
if len(security_alerts) == 0:

    print("\n✅ Device is SAFE → Running Medical AI System...")

    # -----------------------------------------------------
    # Run the medical pipeline (main.py)
    # -----------------------------------------------------
    # main.py already:
    # - loads patient data
    # - preprocesses signals
    # - runs LSTM autoencoder
    # - computes risk scores
    # - generates clinical alerts

    import subprocess

    subprocess.run(["python", "main.py"])


# =========================================================
# CASE 2: DEVICE IS COMPROMISED (ATTACK DETECTED)
# =========================================================
else:

    print("\n🚨 Device is COMPROMISED → Blocking Medical Analysis!")

    # -----------------------------------------------------
    # In real systems:
    # - do NOT trust incoming medical data
    # - raise security alerts
    # - notify SOC / admin
    # -----------------------------------------------------
security_df = pd.DataFrame(security_log)
security_df.to_csv("../../outputs/security_log.csv", index=False)
device_status = "SAFE" if len(security_alerts) == 0 else "COMPROMISED"

print("Original samples:", len(X_new))
print("Safe samples:", len(secure_data_indices))
print("Attack samples:", len(security_alerts))