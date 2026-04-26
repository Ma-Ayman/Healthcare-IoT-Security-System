# ==============================
# STEP 2: DATA PREPROCESSING
# ==============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==============================
# Load merged dataset
# ==============================
df = pd.read_csv("../../data/security/merged_data.csv")

# ==============================
# 1. Handle Labels (Target Column)
# ==============================


# ==============================
# Create a LabelEncoder object
# ==============================
# LabelEncoder is used to convert categorical labels (text)
# into numeric values that machine learning models can understand
le = LabelEncoder()

# ==============================
# Fit + Transform the labels column
# ==============================
# fit_transform does two things:
# 1. Learns all unique classes in the 'label' column (fit)
# 2. Converts each class into a number (transform)
df["label"] = le.fit_transform(df["label"])

# =========================================================
# Display mapping between original attack labels and encoded numbers
# This helps us understand how each attack type is converted into a numeric label
# which is required for machine learning models
# =========================================================

print("Classes mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# SAVE encoder for later use in inference system
import joblib
joblib.dump(le, "../../models/label_encoder.pkl")


# ==============================
# 2. Split Features and Target
# ==============================
X = df.drop("label", axis=1)   # input features
y = df["label"]              # output (attack type)


# ==============================
# 3. Feature Scaling
# ==============================
# Why scaling?
# Because features have different ranges (important for ML models)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Preprocessing done")
# Just confirmation message

# ==============================
# 4. Save processed data
# ==============================


import numpy as np
# Save features after scaling
#np.save("../../data/security/X.npy", X_scaled)

# Save labels (attack types)
#np.save("../../data/security/y.npy", y)

np.savez_compressed(
    "../../data/security/security_dataset.npz",
    X=X_scaled,
    y=y
)