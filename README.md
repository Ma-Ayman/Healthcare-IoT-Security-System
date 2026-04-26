Dataset used:
PhysioNet Challenge 2012 – ICU Mortality Prediction Dataset

Due to size constraints, the dataset is not included in this repository.
It can be accessed from PhysioNet official website.

https://physionet.org/content/challenge-2012/1.0.0/
Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012



Security Dataset:
UNB CIC IoT Dataset (Kaggle)

This dataset contains real IoT network traffic collected from multiple IoT devices under normal and attack scenarios.
It includes different types of cyber attacks such as DDoS, DoS, Brute Force, Reconnaissance, and Mirai botnet.

Link:
https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset

Note: The dataset is not included in the repository due to its large size.




Data Loading Module (data_loader.py)
Purpose
This module is responsible for loading raw patient data from multiple files and combining them into a single unified dataset for further processing in the healthcare IoT pipeline.

Functionality
Reads all CSV files from a given folder containing patient data
Loads each file into a DataFrame
Adds a patient_id column to identify the source patient for each record
Merges all patient datasets into one consolidated DataFrame

How It Works (Step-by-Step)
The system scans the input folder for all patient files.
Each file is read using pandas.read_csv().
A new column patient_id is added to preserve patient identity.
Each patient dataset is stored in a list.
All datasets are concatenated into a single DataFrame.

Input
folder_path: Path containing multiple patient CSV files
Each file represents time-series medical sensor data for one patient.

Output
df_all: A combined DataFrame containing data from all patients with an added patient_id column

Next Step in Pipeline
The output df_all is passed to: preprocessing.py




Data Preprocessing Module (preprocessing.py)
Purpose
This module is responsible for transforming raw ICU patient data into a clean, structured, and machine-learning-ready format.
It prepares the data for time-series modeling by handling formatting, missing values, and feature scaling.

Pipeline Overview

This module performs four main stages:

Data reshaping (Long → Wide format)
Missing data handling
Feature scaling (normalization)
Final dataset preparation
Step 1: Data Reshaping (pivot_data)
What it does
Converts raw long-format data into a structured time-series format.
Each row = one time point
Each column = a physiological parameter
Why it is important
Machine learning models (especially LSTM) require data in structured time-series format instead of raw event-based format.
Output
df_wide: Patient-level time-series dataset indexed by (patient_id, Time)

Step 2: Missing Data Handling (handle_missing)
What it does
Handles missing sensor readings using multiple strategies:
Creates missing value indicators (_missing flags)
Forward fill (time continuity per patient)
Backward fill (initial missing values)
Patient-wise mean imputation
Global fallback (fill remaining NaNs with 0)
 Why it is important
Medical IoT data is often incomplete due to sensor noise or transmission issues.
This step ensures model stability and preserves clinical meaning of missingness.
Output
df_filled: Cleaned dataset
missing_flags: Binary indicators of missing values

Step 3: Feature Scaling (scale_features)
What it does
Applies StandardScaler per patient individually.
Normalizes each patient separately
Preserves individual physiological patterns
Why it is important
Patients have different baseline vitals, so global scaling would distort medical meaning.
Output
scaled_data: Normalized feature dataset

Step 4: Full Preprocessing Pipeline (preprocess_data)
What it does
Combines all preprocessing steps into one pipeline:
Pivot raw data
Handle missing values
Scale features
Combine scaled data with missing indicators
Output
df_final: Fully processed dataset ready for modeling


input df_all (from data_loader.py) - Processing(Cleaning + structuring + scaling) - Output(df_final) - passed to: (windowing.py)

Time-series Windowing Module (windowing.py)
Purpose

This module transforms preprocessed ICU time-series data into fixed-length sliding windows.
These windows are required to train deep learning models such as LSTM Autoencoders.

Pipeline Position

This module comes after: preprocessing.py (df_final) and prepares data for: model.py (LSTM Autoencoder)

What It Does

Raw time-series data is continuous and variable in length.
Deep learning models require fixed-size inputs, so this module:
Splits each patient's time-series data into overlapping sequences
Maintains temporal order
Preserves patient identity for each sequence
Function: create_sequences
Input
df: Preprocessed ICU dataset (df_final)
feature_cols: List of physiological features (e.g., HR, BP, Temp)
window_size: Length of each sequence (default = 6 timesteps)
Process
For each patient:
Extract feature values as a NumPy array
Slide a fixed-size window across time
Generate overlapping sequences
Store:
Sequence data
Corresponding patient ID


The function returns:

X_sequences: NumPy array of shape
(num_samples, window_size, num_features)
patient_ids: Array mapping each sequence to a patient

Input (df_final from preprocessing.py) - Processing(Sliding window segmentation) - Output (X_sequences) - passed to:( model.py)


LSTM Autoencoder Model (model.py)
Purpose
This module defines the core deep learning model used for anomaly detection in ICU time-series data.
The model learns normal physiological behavior and detects anomalies based on reconstruction error.
Pipeline Position
This module comes after: windowing.py (X_sequences) and produces outputs used in: integration.py (decision layer + security system)

Model Idea (How it works)

The system uses an LSTM Autoencoder, which works in two phases:

Encoder: learns compressed representation of normal behavior
Decoder: reconstructs the original input sequence

If reconstruction is poor → the data is abnormal

Model Architecture
Input Layer
Shape: (timesteps, features)
Represents a window of patient physiological signals
Encoder (Compression Stage)
LSTM (64 units) → learns temporal patterns
LSTM (32 units) → compresses into latent space

Output = compressed representation of normal behavior

Bottleneck (Repeat Vector)
Repeats compressed vector across time steps
Prepares data for reconstruction
Decoder (Reconstruction Stage)
LSTM (32 units)
LSTM (64 units)
Reconstructs original time-series structure

Output Layer
TimeDistributed Dense layer
Reconstructs original feature space at each timestep
Learning Objective
The model is trained using:
Loss Function: Mean Squared Error (MSE)
Interpretation:
Low reconstruction error → normal behavior
High reconstruction error → anomaly
Output of Model

The model does not directly classify data.
Instead it produces: Reconstructed sequence
Reconstruction error (used later for anomaly detection)

Input (X_sequences from windowing.py) - Processing(Learn normal ICU behavior patterns) - Output(Reconstruction model)- passed to:(integration.py (for risk + security decisions))


Main Pipeline (main.py)
Purpose
This script represents the end-to-end execution pipeline of the AI-powered ICU monitoring system.
It connects all modules together to simulate a real-world healthcare IoT workflow, starting from raw patient data and ending with anomaly detection, risk scoring, and clinical decision outputs.

Pipeline Overview
The system follows a sequential workflow: Data Loading → Preprocessing → Windowing → Model Training → Anomaly Detection → Risk Scoring → Clinical Alerts → Saving Results


Step 1: Load Raw IoT Data (data_loader.py)
The pipeline starts by loading raw ICU patient data from multiple CSV files.
Key actions:
Reads all patient files from a folder
Combines them into one dataset
Adds patient_id to preserve identity
Output:A unified dataset containing all patients: df_all
Next Step:This raw dataset is passed to the preprocessing module.

Step 2: Data Preprocessing (preprocessing.py)
Raw ICU data is cleaned and transformed into a structured machine learning format.
Key operations:
Converts data from long → wide format
Handles missing values (forward fill, backward fill, imputation)
Adds missing value indicators
Applies patient-wise normalization
Output: A clean, structured dataset:df_final
Next Step: This processed data is passed to the windowing module.

Step 3: Time-Series Windowing (windowing.py)
Since deep learning models require fixed-size input, the time-series data is split into sliding windows.
Key operations:
Splits each patient’s timeline into sequences
Maintains temporal order
Associates each sequence with a patient ID
Output:X → sequences of shape (samples, timesteps, features) , pids → patient mapping for each sequence
Next Step: These sequences are used to train the deep learning model.

Step 4: Model Building (model.py)
An LSTM Autoencoder is built to learn normal physiological behavior.
Model role: Learns patterns of normal ICU signals , Reconstructs input sequences , Detects anomalies via reconstruction error
Output:A compiled LSTM Autoencoder model ready for training.
Next Step: The model is trained using the generated sequences.

Step 5: Model Training
The model is trained in an unsupervised manner.
Key idea:
Input = Output (autoencoder training)
The model learns to reconstruct normal patterns
Result: After training: The model becomes sensitive to abnormal behavior , High reconstruction error indicates anomalies

Step 6: Reconstruction & Error Calculation
After training, the model reconstructs the input sequences.
Key operation:
Compare original vs reconstructed signals
Compute reconstruction error (MSE)
Output: Each sequence gets a loss value

Step 7: Patient-Specific Thresholding
Instead of using a global threshold, the system computes a personal baseline for each patient.
Key idea:
Each patient has unique physiological behavior
Threshold = 95th percentile of their own reconstruction error
Output:Threshold per patient Used to detect anomalies more accurately

Step 8: Anomaly Detection
Each sequence is classified based on deviation from patient baseline.
Logic:
If loss > threshold → anomaly , Otherwise → normal
Output:Binary anomaly labels per time window

Step 9: Risk Scoring
A continuous risk score is calculated.
Meaning:Measures severity of deviation
Higher score = more critical condition

Step 10: Clinical Alert Generation
Risk scores are converted into interpretable clinical alerts:
NORMAL → stable condition , WARNING → needs monitoring , CRITICAL → urgent attention required

Step 11: Patient-Level Analysis
The system aggregates window-level predictions into patient-level insights.
Key operations:
Average risk per patient , Ranking patients by severity
Output: ICU priority list , High-risk patient identification

Step 12: Results Saving
Final outputs are saved for visualization and dashboard integration.
Saved files:window-level predictions , patient risk scores

The main pipeline integrates all modules into a complete AI-driven ICU monitoring system. It simulates real-world healthcare IoT environments by continuously processing patient data, detecting anomalies using deep learning, and generating risk-aware clinical alerts.









Security Module (Cybersecurity Layer)
Overview
This module represents the first line of defense in the Healthcare IoT system.
Before any medical analysis is performed, the system first ensures that incoming IoT data is safe and not affected by cyber attacks or malicious traffic.
This layer simulates a real-world Intrusion Detection System (IDS) for healthcare environments.

Data Collection (data_security folder)
data_security/
This folder contains multiple CSV files representing: IoT network traffic , Attack scenarios , Normal and malicious patterns
These datasets simulate real healthcare IoT security environments.

Data Loading Module (load_data_security.py)
This script is responsible for loading and merging all cybersecurity datasets into one unified structure.

Step-by-step functionality:
1. Define dataset path
The system points to a folder containing multiple CSV files representing different attack scenarios.
2. File discovery
All CSV files inside the folder are automatically detected using pattern matching.
This allows scalability (easy to add new attack data)
3. Data loading
Each file is: Read into a pandas DataFrame and Stored in a temporary list
4. Data merging
All individual datasets are combined into one unified dataset:final_df

This ensures that:mNormal traffic + attack data are in one structure Ready for ML training or inference
Output of this module The module produces: merged dataset: final_df

This dataset contains:All attack types , Normal traffic samples , Unified feature space
Final Step in this module The merged dataset is saved as: merged_data.csv


Security Data Preprocessing Module (preprocessing_security.py)
Purpose
This module is responsible for preparing the merged cybersecurity dataset for machine learning.
It transforms raw network traffic data into a numerical, normalized, and model-ready format that can be used for attack detection.

Pipeline Position
This module comes after: load_data_security.py (merged_data.csv) and prepares data for: Security Model (Attack Detection - XGBoost or classifier)
The preprocessing pipeline consists of four main steps:
Label Encoding (convert attack types to numbers)
Feature/Target separation
Feature scaling (normalization)
Saving processed data for model training
Step 1: Label Encoding (Attack Classification)
The system converts categorical attack labels (text) into numerical values.
Example: BenignTraffic → 0 , DoS → 1 , Botnet → 2
Why it is important : Machine learning models cannot understand text labels, so they must be converted into numbers.
The mapping between:Original attack names and Encoded numeric values is printed and saved.
This is important for: Model interpretability and Inference stage later
Saved Output : A trained encoder is saved as: label_encoder.pkl This will be reused later in the inference stage to decode predictions.

Step 2: Feature and Target Separation
The dataset is split into: X (features) → network traffic attributes and y (labels) → attack type
Why it is important : This separation is required for supervised machine learning models.

Step 3: Feature Scaling
All input features are normalized using StandardScaler.
Why it is important : Features in network traffic have different ranges
Scaling ensures: Stable model training , Better performance and Faster convergence
Output : Scaled feature matrix: X_scaled

Step 4: Saving Processed Data
The processed dataset is saved for later use in the security model.
Saved Files: X.npy   → Scaled input features and y.npy   → Encoded attack labels

Security Model Training (train_model.py)
Purpose: This module is responsible for training a machine learning model to detect cyber attacks in healthcare IoT traffic.
It learns patterns from the preprocessed dataset and builds a classifier capable of distinguishing between: Normal traffic (Benign) and Different types of cyber attacks
Pipeline Position
This module comes after: preprocessing_security.py (X.npy, y.npy) and produces output used in: integration.py (real-time attack detection)
What This Module Does: The training pipeline consists of the following stages:
Load processed data
Split into training/testing sets
Build the model (XGBoost)
Handle class imbalance
Train the model
Evaluate performance
Save trained model


Step 1: Load Preprocessed Data
The system loads previously saved NumPy arrays: X → input features (scaled network data) and y → encoded attack labels
Why it is important : This ensures: Consistency between preprocessing and training and Faster pipeline execution (no reprocessing needed)


Step 2: Train-Test Split
What happens : The dataset is divided into: Training set (80%) → used to train the model and Testing set (20%) → used to evaluate performance
Why it is important : This allows us to measure how well the model generalizes to unseen data.

Step 3: Model Creation (XGBoost Classifier)
What happens : An XGBoost Classifier is initialized.
Why XGBoost : High performance on tabular data , Handles complex patterns , Robust to noise and Suitable for multi-class classification
Model Objective
The model performs: Multi-class classification and predicts attack type (not just attack vs normal)

Step 4: Handling Class Imbalance 
Problem : Cybersecurity datasets are usually imbalanced: Normal traffic → very frequent , Some attacks → very rare
Solution : The system calculates class weights: Rare classes → higher weight and Frequent classes → lower weight
How it works: Each training sample is assigned a weight based on its class frequency. This tells the model: "Pay more attention to rare attacks"
Why it is important : Without this step: Model may ignore rare but critical attacks and Accuracy might look high but model is weak


Step 5: Model Training
The model is trained using: Input: X_train , Labels: y_train and Sample weights: to handle imbalance
The model learns: Patterns of normal traffic and Patterns of different attack types

Step 6: Model Evaluation
The model is tested on unseen data (X_test).
Metrics used:Accuracy(Overall correctness of predictions) , Classification Report Includes:Precision , Recall , F1-score for each attack type
Why it is important : Provides detailed insight into: Which attacks are detected well , Which attacks need improvement


Step 7: Save Trained Model
The trained model is saved as:xgboost_model.pkl
Why it is important: This allows: Reusing the model without retraining , Deploying it in real-time systems and Using it in integration pipeline

This module transforms cybersecurity data into an intelligent detection system capable of identifying multiple types of attacks in healthcare IoT environments.


Security Inference Module (security_inference.py)
This module represents the deployment stage of the cybersecurity system.
It is responsible for taking new incoming IoT data and determining whether it is: Normal traffic (Benign) Or a cyber attack (e.g., DDoS, MITM, etc.)
This module comes after: train_model.py (xgboost_model.pkl + label_encoder.pkl) and is used in: Integration Layer (final system combining medical + security decisions)

What This Module Does: This module simulates a real-time cybersecurity monitoring system.
It performs the following:
Load trained model
Load label encoder
Receive new incoming data
Predict attack type
Convert predictions to readable labels
Return final decision

Step 1: Load Trained Model
The system loads the trained XGBoost model from disk.
Why it is important : Avoids retraining every time , Enables real-time deployment and Ensures consistency with training phase

Step 2: Load Label Encoder
The same LabelEncoder used during preprocessing is loaded.
Why it is important : During training: Labels were converted from text → numbers Now: We must convert predictions back from numbers → readable labels

Step 3: Receive New IoT Data
Input : X_new → new incoming data from IoT devices , Format: NumPy array , Same structure as training data 
Real-world meaning
This represents: Network packets , Sensor communication data and Device behavior logs

Step 4: Model Prediction
The model predicts a class for each input sample.
Output (before decoding) : Numeric labels like:0 1 2
Meaning : Each number corresponds to a specific attack type.

tep 5: Convert Predictions to Readable Labels
The system uses the LabelEncoder to convert numbers into actual attack names.
Example Output : Instead of: [0, 2, 1] We get: ["BenignTraffic", "DDoS", "MITM"]
Why it is important : Makes output human-readable , Enables integration with alert systems and Useful for dashboards and reports

Step 6: Return Results
The function returns: Array of attack labels , One label per input sample

This module transforms a trained machine learning model into a real-time cybersecurity detection system capable of analyzing incoming IoT data and identifying potential attacks instantly.


System Integration Module (integration.py)
This module represents the core intelligence layer of the system.
It integrates two main components:
Cybersecurity Model (XGBoost) → detects malicious IoT traffic
Medical AI Model (LSTM Autoencoder) → detects patient anomalies and risk
Why This Module Is Critical Without this module: The system would consist of two disconnected parts and No real decision-making process exists

With this module: The system becomes a complete intelligent healthcare IoT pipeline and It ensures that only trusted data is used for medical decisions
Medical analysis must never be performed on untrusted data
This module enforces a strict pipeline: Validate data security Then perform medical analysis

Step 1: Simulated IoT Data Input
Loads a subset of data (X.npy): Simulates incoming real-time IoT sensor data
Purpose : In real-world systems: Data arrives continuously from devices
In this implementation: A dataset is used to simulate streaming behavior

Step 2: Security Check 
Incoming data is passed to: detect_attacks()
Output: Each sample is classified as: BenignTraffic → Safe or Attack types (e.g., DDoS, MITM, etc.) → Malicious
Importance
This is the first line of defense: Prevents compromised data from entering the medical pipeline and Ensures system reliability

Step 3: Security Decision Logic
Each sample is evaluated and categorized:
Case 1: Attack Detected : A security alert is generated , Sample is marked as COMPROMISED and Excluded from further processing
Case 2: Safe Data : Sample is marked as SAFE and passed to the medical system
Outputs:
security_alerts → list of detected attacks
secure_data_indices → indices of safe samples
security_log → full tracking log

Step 4: Data Filtering
Only safe data is retained: Raw Data → Security Filter → Safe Data Only
Ensures: No corrupted or malicious data reaches the medical model and Maintains integrity of medical decisions

Step 5: System-Level Decision
This is the critical decision point of the entire system.

Case 1: System is SAFE
Condition : No attacks detected
Action : Run the medical pipeline (main.py) Meaning: Data is trusted and Medical analysis can proceed safely

Case 2: System is COMPROMISED
Condition : One or more attacks detected
Action : Block medical analysis and Raise security alerts
Meaning : Data cannot be trusted and Prevents incorrect clinical decisions


Step 6: Security Logging
All events are stored in: security_log.csv
Contents : Each record includes: sample_id , traffic_type (NORMAL / ATTACK) , attack_type and status (SAFE / COMPROMISED)
Purpose : Enables monitoring and auditing , Supports dashboard visualization and Useful for security analysis


ICU Monitoring Dashboard (dashboard.py)
This module represents the final layer of the system, where all AI outputs are transformed into a visual, user-friendly clinical dashboard.

It integrates both:
Security Monitoring Layer (Cyber Attack Detection) and  Medical Monitoring Layer (Patient Risk & Anomaly Detection)

The dashboard simulates a real-world ICU interface, allowing clinicians to:
Monitor patient conditions in real-time
Detect anomalies early
Prioritize high-risk patients
Ensure incoming IoT data is secure and trustworthy

System Role in the Pipeline

The dashboard consumes outputs from multiple modules:
main.py → generates: window_level_results.csv (risk scores, alerts)
integration.py → generates:security_log.csv (attack detection results)

These outputs are visualized and interpreted in a clinical context.

Security Monitoring Layer
This is the first section of the dashboard, reflecting the system’s cybersecurity status.
In real-world IoT healthcare systems: Data streams from sensors are validated before use , Malicious data is blocked immediately
Key Metrics
Total Processed Samples : Number of incoming IoT data samples analyzed
Total Attacks Detected : Number of malicious events identified by the security model
Stream Status : SAFE → Data is trusted , COMPROMISED → Potential cyber attack detected

Security Logs
Displays all analyzed samples Each entry includes:Sample ID , Traffic Type (Normal / Attack) , Attack Type and Status (SAFE / COMPROMISED)

Security Decision Impact
The system clearly indicates: If compromised → Medical analysis may be restricted , If safe → Data is allowed for medical processing
This reflects a zero-trust pipeline design

Medical Monitoring Layer
This section represents the clinical AI system for ICU monitoring.
It provides both: Real-time patient monitoring , System-wide insights , ICU Overview Metrics (KPIs)
Total Patients → Number of monitored ICU patients
Total Windows → Number of analyzed time segments
Detected Anomalies → Abnormal physiological patterns
Critical Alerts → High-risk medical situations

These metrics provide a quick snapshot of ICU status.

Patient-Level Monitoring
Patient Selection : Users can select a specific patient and then The dashboard dynamically updates all visualizations

Real-Time Patient Status
Based on the latest risk score:
CRITICAL → Immediate intervention required
WARNING → Close monitoring needed
NORMAL → Stable condition

Time-Series Risk Visualization
This section shows how patient condition evolves over time.
Features:Continuous risk score curve , Highlighted anomalies
Clinical thresholds:Warning threshold and Critical threshold
Insight:Spikes above thresholds indicate abnormal physiological behavior.

Clinical Alert Log
Displays recent AI-generated alerts for the selected patient.
Each entry includes: Patient ID , Risk Score and Alert Level (NORMAL / WARNING / CRITICAL)

Color coding improves readability: Red → Critical , Orange → Warning , Green → Normal

ICU-Wide System Insights
Alert Distribution: Shows distribution of alert types across all patients it Helps assess overall ICU pressure and severity

High-Risk Patient Prioritization
Patients are ranked based on: Average risk score over time
Purpose: Simulates ICU triage system and Helps clinicians prioritize critical cases

The dashboard transforms the system into a complete AI-powered healthcare solution by:

Combining cybersecurity + medical intelligence
Enabling real-time monitoring
Supporting clinical decision-making
Improving patient safety and system reliability