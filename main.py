import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")

st.title("Early Identification of Learning Gaps & Dropout Risk")

# -------------------------
# Sample student dataset
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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# User input
# -------------------------
st.sidebar.header("Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
marks = st.sidebar.slider("Internal Marks", 0, 100, 60)
assignment = st.sidebar.slider("Assignment Score", 0, 100, 65)
participation = st.sidebar.slider("Participation (1–10)", 1, 10, 5)

input_data = np.array([[attendance, marks, assignment, participation]])

# Clearer prediction output
if risk == "High":
    st.write("Prediction Result: High Dropout Risk — Early intervention recommended.")
else:
    st.write("Prediction Result: Low Dropout Risk — Student likely to continue successfully.")
# Recommendations based on risk
if risk == "High":
    st.write("Recommendations:")
    st.write("- Assign academic mentor")
    st.write("- Extra remedial classes")
    st.write("- Regular attendance monitoring")
    st.write("- Counseling session")
else:
    st.write("Recommendations: Continue regular monitoring and encourage participation.")
# -------------------------
# Output
# -------------------------
st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ High Dropout Risk (Risk Score: {risk:.2f})")
else:
    st.success(f"✅ Low Dropout Risk (Risk Score: {risk:.2f})")

st.markdown("---")
st.caption("Hackathon Demo | ML-based Early Warning System")