# -------------------------
# IMPORT LIBRARIES
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# STREAMLIT PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")

st.title("Early Identification of Learning Gaps & Dropout Risk")

# -------------------------
# SAMPLE STUDENT DATA
# -------------------------
data = {
    "Attendance (%)": [95, 60, 85, 40, 70, 55, 90, 30],
    "Internal Marks": [88, 45, 75, 30, 65, 50, 92, 25],
    "Assignment Score": [90, 50, 80, 35, 70, 55, 95, 20],
    "Participation": [8, 4, 7, 2, 6, 5, 9, 1],
    "Dropout": [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# Train model (RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# USER INPUTS
# -------------------------
st.sidebar.header("Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
marks = st.sidebar.slider("Internal Marks", 0, 100, 60)
assignment = st.sidebar.slider("Assignment Score", 0, 100, 65)
participation = st.sidebar.slider("Participation (1–10)", 1, 10, 5)

input_data = np.array([[attendance, marks, assignment, participation]])

# -------------------------
# PREDICTION & 3-LEVEL RISK
# -------------------------
risk_score = model.predict_proba(input_data)[0][1]  # probability of dropout

# Define risk levels
if risk_score < 0.3:
    risk = "Low"
elif risk_score < 0.7:
    risk = "Medium"
else:
    risk = "High"

# Display prediction
st.subheader("Prediction Result")
if risk == "High":
    st.error(f"⚠️ High Dropout Risk — Early intervention recommended (Risk Score: {risk_score:.2f})")
elif risk == "Medium":
    st.warning(f"⚠️ Medium Dropout Risk — Monitor closely (Risk Score: {risk_score:.2f})")
else:
    st.success(f"✅ Low Dropout Risk — Student likely to continue successfully (Risk Score: {risk_score:.2f})")

# -------------------------
# RECOMMENDATIONS
# -------------------------
st.subheader("Recommendations")
if risk == "High":
    st.write("- Assign academic mentor")
    st.write("- Extra remedial classes")
    st.write("- Regular attendance monitoring")
    st.write("- Counseling session")
elif risk == "Medium":
    st.write("- Provide guidance sessions")
    st.write("- Monitor performance regularly")
    st.write("- Encourage participation in class activities")
else:
    st.write("- Continue regular monitoring")
    st.write("- Encourage participation in class activities")

# -------------------------
# DISCLAIMER
# -------------------------
st.markdown("---")
st.caption("Disclaimer: This is an early-warning prototype. Predictions depend on input data and do not cover personal or financial factors.")
