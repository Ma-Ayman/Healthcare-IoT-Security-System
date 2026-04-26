# ==============================
# STEP 3: MODEL TRAINING
# ==============================

import numpy as np
# Used to load saved arrays (X.npy, y.npy)

from sklearn.model_selection import train_test_split
# Used to split data into training and testing sets

from sklearn.metrics import classification_report, accuracy_score
# Used to evaluate model performance

from xgboost import XGBClassifier
# The machine learning model we will use

import joblib
# Used to save trained model


# ==============================
# 1. LOAD PREPROCESSED DATA
# ==============================

#X = np.load("data/security/X.npy")
# Load features after scaling

#y = np.load("data/security/y.npy")
# Load labels (attack types)

data = np.load("../../data/security/security_dataset.npz")

X = data["X"]
y = data["y"]

# ==============================
# 2. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    # 20% of data for testing
    random_state=42
    # ensures same split every time
)

# ==============================
# 3. CREATE MODEL
# ==============================

model = XGBClassifier(
    n_estimators=100,
    # number of trees

    learning_rate=0.1,
    # how fast model learns

    max_depth=6,
    # depth of trees (controls complexity)

    objective="multi:softmax"
    # because we have MULTI-class classification (many attack types)
)
# =========================================================
# HANDLE CLASS IMBALANCE USING CLASS WEIGHTS
# =========================================================


from collections import Counter
# Counter helps us count how many samples exist for each class

# =========================================================
# COUNT HOW MANY SAMPLES IN EACH CLASS
# =========================================================

# y_train contains labels (attack types)
# we count how many times each attack appears
class_counts = Counter(y_train)

# Example output:
# {0: 50, 1: 10000, 2: 300}
# meaning:
# class 0 is rare
# class 1 is very frequent

print("Class distribution in training data:")
print(class_counts)

# total number of training samples
total_samples = len(y_train)

# number of unique classes (how many attack types we have)
num_classes = len(class_counts)

# dictionary to store weight for each class
class_weights = {}

# loop over each class and its count
for cls, count in class_counts.items():

    # -----------------------------------------------------
    # IDEA:
    # If a class appears a lot → give it small weight
    # If a class appears rarely → give it big weight
    #
    # Formula used:
    # weight = total_samples / (num_classes * class_count)
    # -----------------------------------------------------

    weight = total_samples / (num_classes * count)

    class_weights[cls] = weight

# print final weights for each attack class
print("\nClass weights calculated:")
print(class_weights)

# =========================================================
# CREATE WEIGHT FOR EACH TRAINING SAMPLE
# =========================================================

# now we assign a weight to every row in y_train
# so each sample knows how important it is


sample_weights = np.array([
    class_weights[label]  # get weight of that class
    for label in y_train  # loop over all training labels
])
print("\nSample weights created successfully")
print("sample weights:", sample_weights)

# ==============================
# 4. TRAIN MODEL
# ==============================
# when training model, we pass sample_weights:

# model.fit(X_train, y_train, sample_weight=sample_weights)

# this tells the model:
# "pay more attention to rare attacks"
# "don't ignore minority classes"

# =========================================================

model.fit(X_train, y_train , sample_weight=sample_weights)
# Model learns patterns between features and attack types


# ==============================
# 5. PREDICTIONS
# ==============================

y_pred = model.predict(X_test)
# Model predicts attack types for unseen data


# ==============================
# 6. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)
# Calculate how many predictions were correct

print("Accuracy:", accuracy)
# Show accuracy result

print("\nClassification Report:\n")
# Detailed performance for each attack type

print(classification_report(y_test, y_pred))

# ==============================
# 7. SAVE MODEL
# ==============================

joblib.dump(model, "../../models/xgboost_model.pkl")
# Save trained model to reuse later

print("Model saved successfully!")

