# =========================================================
# IMPORT LIBRARIES
# =========================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



# =========================================================
# PAGE CONFIG 
# =========================================================
st.set_page_config(page_title="ICU Dashboard", layout="wide")


# =========================================================
# CUSTOM UI STYLE
# =========================================================
st.markdown("""
<style>

/* ===== MAIN TEXT ===== */
html, body, [class*="css"]  {
    color: white !important;
}

/* ===== KPI CARDS ===== */
[data-testid="stMetric"] {
    background-color: #1C2541;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #3A506B;
}

/* ===== KPI VALUE ===== */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;   
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* ===== KPI LABEL ===== */
[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 14px !important;
}

/* ===== TITLE ===== */
h1 {
    color: #5BC0BE !important;
}

/* ===== SUBTITLE ===== */
h2, h3 {
    color: #CDE8E5 !important;
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# LOAD MODEL OUTPUTS 
# =========================================================

# This dataset is generated from the AI pipeline (LSTM Autoencoder)
# It contains:
# - patient_id: identifier for each ICU patient
# - risk_score: anomaly severity score
# - anomaly: binary flag (normal / abnormal)
# - alert: clinical decision (NORMAL / WARNING / CRITICAL)

df = pd.read_csv("../outputs/window_level_results.csv")

# =========================================================
# SECURITY LAYER DASHBOARD (NEW SECTION)
# =========================================================
# This section displays results from the Security Model (XGBoost)
# It shows whether IoT sensor streams are safe or compromised
#
# In real systems:
# - This is the FIRST layer of defense
# - It monitors incoming IoT traffic
# - It blocks malicious streams before reaching medical system
# =========================================================

import pandas as pd
import streamlit as st

# =========================================================
# LOAD SECURITY OUTPUT DATA
# =========================================================
# This file is generated from security_inference / integration module
# It contains:
# - sample_id (time/window index)
# - attack_type (if any)
# - status (ATTACK / SAFE)
# =========================================================

security_df = pd.read_csv("../outputs/security_log.csv")


# =========================================================
# SECURITY OVERVIEW METRICS
# =========================================================

st.subheader("🛡️ Security Monitoring Layer")

import numpy as np

data = np.load("../Data/security/security_dataset.npz")
X_new = data["X"][:100]
total_samples = len(X_new)

# Count how many attacks were detected
total_attacks = len(security_df[security_df["attack_type"] != "BenignTraffic"])

# Stream status (stream-level decision)
stream_status = "SAFE" if total_attacks == 0 else "COMPROMISED"


# =========================================================
# DISPLAY KPI CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("Total Processed Samples", total_samples)
col2.metric("Total Attacks Detected", total_attacks)
col3.metric("Stream Status", stream_status)


# =========================================================
# SECURITY ALERT LOG
# =========================================================

st.subheader("🚨 Security Logs")




attack_logs = security_df[security_df["traffic_type"] == "ATTACK"]
normal_logs = security_df[security_df["traffic_type"] == "NORMAL"]
st.dataframe(security_df)



# =========================================================
# SECURITY IMPACT MESSAGE (VERY IMPORTANT)
# =========================================================

if stream_status == "COMPROMISED":
    st.error(
        "⚠️ WARNING: IoT data stream is compromised. "
        "Medical system may restrict or ignore incoming data."
    )
else:
    st.success(
        "✅ IoT stream is trusted. Data is safe for medical analysis."
    )



# =========================================================
# DASHBOARD TITLE (UI LAYER)
# =========================================================

# This defines the main title of the ICU monitoring system
# It represents the user-facing clinical interface
st.title("🧠 AI-Powered ICU Monitoring Dashboard")

# Subheader explains the system purpose clearly for clinicians
st.subheader("Real-Time ICU Monitoring System (AI-Based Anomaly Detection & Risk Scoring)")

# =========================================================
# OVERVIEW SECTION (GLOBAL ICU STATUS)
# =========================================================

# This section provides a high-level summary of ICU state
# It helps clinicians quickly understand overall system health

total_patients = df["patient_id"].nunique()   # number of unique ICU patients monitored
total_windows = len(df)                       # total number of time windows processed
total_anomalies = df["anomaly"].sum()         # total detected abnormal windows
critical_cases = (df["alert"] == "CRITICAL").sum()  # number of high-risk alerts




# =========================================================
# DISPLAY KEY PERFORMANCE INDICATORS (KPIs)
# =========================================================

# These metrics simulate ICU monitoring dashboard cards
# They provide quick insight without needing detailed analysis

col1, col2, col3, col4 = st.columns(4)


st.markdown("""
<div style='font-size:18px; line-height:1.6'>
📊 ICU Overview Explanation

- Total Patients: Total number of ICU patients monitored by the system  
- Total Windows: Number of time segments analyzed by the AI model  
- Detected Anomalies: Number of abnormal patterns detected  
- Critical Alerts: High-risk situations requiring immediate attention  

 These metrics provide a quick summary of the ICU status.
</div>
""", unsafe_allow_html=True)


col1.metric("Total Patients", total_patients)
col2.metric("Total Windows", total_windows)
col3.metric("Detected Anomalies", total_anomalies)
col4.metric("Critical Alerts", critical_cases)


# =========================================================
# PATIENT SELECTION MODULE
# =========================================================

# This allows clinicians to select a specific patient
# and inspect their physiological behavior over time

patient_id = st.selectbox(
    "Select Patient for Detailed Monitoring",
    sorted(df["patient_id"].unique())
)

# Filter dataset to only selected patient
# This simulates patient-specific ICU monitoring view
patient_df = df[df["patient_id"] == patient_id]

# ============================
# REAL-TIME PATIENT STATUS
# ============================

current_risk = patient_df["risk_score"].iloc[-1]

if current_risk > 1.2:
    st.error("🚨 CRITICAL CONDITION - Immediate Action Required")
elif current_risk > 1.0:
    st.warning("⚠️ WARNING - Monitor Closely")
else:
    st.success("✅ NORMAL - Stable Condition")




# =========================================================
# TIME-SERIES RISK VISUALIZATION
# =========================================================

st.subheader("📈 Patient Risk Score Over Time")

# Create figure for plotting risk evolution
fig, ax = plt.subplots()

# Plot risk score across time windows
# Each point represents model output for a specific time window
# Plot risk line
ax.plot(
    patient_df.index,
    patient_df["risk_score"],
    color="#5BC0BE",   # soft teal
    linewidth=2,
    label="Risk Score"
)

# Highlight anomalies 
anomalies = patient_df[patient_df["anomaly"] == True]
ax.scatter(
    anomalies.index,
    anomalies["risk_score"],
    color="#FF6B6B",
    s=60,
    label="Anomalies"
)

# Compute dynamic threshold for visualization (mean + std)
# This represents expected normal range boundary
# More realistic clinical thresholds
ax.axhline(1.0, color="#FFD166", linestyle="--", linewidth=2, label="Warning")
ax.axhline(1.2, color="#EF476F", linestyle="--", linewidth=2, label="Critical")



# Label axes for interpretability
ax.set_xlabel("Time Window")
ax.set_ylabel("Risk Score")
ax.set_title("Risk Evolution Over Time", color="white")
ax.legend()
ax.grid(alpha=0.3)
# Render plot inside dashboard
fig.patch.set_facecolor('#1C2541')
ax.set_facecolor('#1C2541')
st.pyplot(fig)



st.markdown("""
<div style='font-size:18px; line-height:1.6'>
Peaks above thresholds indicate abnormal physiological behavior requiring attention.
</div>
""", unsafe_allow_html=True)
# =========================================================
# ALERT COLORING FUNCTION
# =========================================================
def color_alert(val):
    if val == "CRITICAL":
        return "background-color: red; color: white"
    elif val == "WARNING":
        return "background-color: orange; color: black"
    else:
        return "background-color: green; color: white"

        
# =========================================================
# REAL-TIME ALERT LOG (CLINICAL DECISIONS)
# =========================================================

st.subheader("🚨 Latest Clinical Alerts")

# Display most recent model decisions for selected patient
# This simulates ICU monitoring alerts received by clinicians

latest = patient_df[["patient_id", "risk_score", "alert"]].tail(10)

styled = latest.style.map(color_alert, subset=["alert"])

st.dataframe(styled)

st.markdown("""
<div style='font-size:18px; line-height:1.6'>
Shows the most recent AI-generated alerts for the selected patient.
</div>
""", unsafe_allow_html=True)


# =========================================================
# ICU-WIDE ALERT DISTRIBUTION
# =========================================================

st.subheader("🏥 ICU System Status Overview")

# Shows distribution of all alerts across ICU patients
# Helps clinicians understand overall ICU load and severity

alert_counts = df["alert"].value_counts()

st.bar_chart(alert_counts)

st.markdown("""
<div style='font-size:18px; line-height:1.6'>
Displays distribution of alert types across all ICU patients.
</div>
""", unsafe_allow_html=True)
# =========================================================
# HIGH-RISK PATIENT PRIORITIZATION
# =========================================================

st.subheader("🔥 High-Risk Patient Ranking")

# Aggregate risk scores per patient
# This converts window-level predictions into patient-level severity

top_risk_patients = (
    df.groupby("patient_id")["risk_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# Display prioritized patients for clinical attention
st.dataframe(top_risk_patients.to_frame().style.background_gradient(cmap="Reds"))

st.markdown("""
<div style='font-size:18px; line-height:1.6'>
Patients are ranked based on their average risk score to prioritize critical cases.
</div>
""", unsafe_allow_html=True)

