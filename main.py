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

attendance = st.sidebar.slider(
    "Attendance (%)", 0, 100, 75,
    help="Percentage of classes attended by the student"
)
marks = st.sidebar.slider(
    "Internal Marks", 0, 100, 60,
    help="Marks obtained in internal exams"
)
assignment = st.sidebar.slider(
    "Assignment Score", 0, 100, 65,
    help="Average score for assignments"
)
participation = st.sidebar.slider(
    "Participation (1–10)", 1, 10, 5,
    help="Class participation rating from 1 (low) to 10 (high)"
)

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

# -------------------------
# DISPLAY PREDICTION
# -------------------------
st.subheader("Prediction Result")
if risk == "High":
    st.error(f"⚠️ High Dropout Risk — Early intervention recommended (Risk Score: {risk_score:.2f})")
elif risk == "Medium":
    st.warning(f"⚠️ Medium Dropout Risk — Monitor closely (Risk Score: {risk_score:.2f})")
else:
    st.success(f"✅ Low Dropout Risk — Student likely to continue successfully (Risk Score: {risk_score:.2f})")

# -------------------------
# DISPLAY PROGRESS BAR
# -------------------------
st.subheader("Dropout Risk Score")
st.write(f"Risk Probability: {risk_score:.2f}")
st.progress(int(risk_score * 100))

# -------------------------
# RECOMMENDATIONS (COLOR-CODED)
# -------------------------
st.subheader("Recommendations")
if risk == "High":
    st.markdown("<span style='color:red'>- Assign academic mentor</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:red'>- Extra remedial classes</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:red'>- Regular attendance monitoring</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:red'>- Counseling session</span>", unsafe_allow_html=True)
elif risk == "Medium":
    st.markdown("<span style='color:orange'>- Provide guidance sessions</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:orange'>- Monitor performance regularly</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:orange'>- Encourage participation in class activities</span>", unsafe_allow_html=True)
else:
    st.markdown("<span style='color:green'>- Continue regular monitoring</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:green'>- Encourage participation in class activities</span>", unsafe_allow_html=True)

# -------------------------
# BAR CHART OF INPUTS
# -------------------------
st.subheader("Student Performance Overview")
perf_df = pd.DataFrame({
    "Metric": ["Attendance", "Internal Marks", "Assignment Score", "Participation"],
    "Value": [attendance, marks, assignment, participation]
})
st.bar_chart(perf_df.set_index("Metric"))

# -------------------------
# DISCLAIMER
# -------------------------
st.markdown("---")
st.caption("Disclaimer: This is an early-warning prototype. Predictions depend on input data and do not cover personal or financial factors.")
