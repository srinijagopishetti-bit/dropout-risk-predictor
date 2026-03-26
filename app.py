import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# TITLE
# -----------------------------
st.set_page_config(page_title="Dropout Predictor", layout="centered")

st.title("🎓 Student Dropout Risk Prediction & Analysis System")

st.write("This system predicts student dropout risk and provides insights and recommendations.")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📥 Enter Student Details")

attendance = st.slider("Attendance (%)", 0, 100, 75)
marks = st.slider("Internal Marks (%)", 0, 100, 60)
assignment = st.slider("Assignment Score (%)", 0, 100, 65)
participation = st.slider("Participation (1–10)", 1, 10, 5)

# -----------------------------
# SIMPLE PREDICTION LOGIC
# (Replace with your ML model if needed)
# -----------------------------
def predict_risk(attendance, marks, assignment, participation):
    score = (100 - attendance) + (100 - marks) + (100 - assignment) + (10 - participation)*5
    
    if score > 180:
        return "High", 80
    elif score > 120:
        return "Medium", 50
    else:
        return "Low", 20

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🔍 Predict Risk"):

    result, risk_score = predict_risk(attendance, marks, assignment, participation)

    # -----------------------------
    # RESULT
    # -----------------------------
    st.subheader("🎯 Prediction Result")

    if result == "High":
        st.error(f"🔴 High Dropout Risk (Score: {risk_score}%)")
    elif result == "Medium":
        st.warning(f"🟡 Medium Dropout Risk (Score: {risk_score}%)")
    else:
        st.success(f"🟢 Low Dropout Risk (Score: {risk_score}%)")

    # -----------------------------
    # PROGRESS BAR
    # -----------------------------
    st.subheader("📊 Dropout Risk Score")
    st.progress(risk_score)

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    st.subheader("📊 Key Factors Affecting Prediction")

    st.write(f"- Attendance impact: {'High' if attendance < 60 else 'Low'}")
    st.write(f"- Marks impact: {'High' if marks < 50 else 'Moderate'}")
    st.write(f"- Assignment impact: {'High' if assignment < 50 else 'Moderate'}")
    st.write(f"- Participation impact: {'High' if participation < 4 else 'Low'}")

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    st.subheader("📈 Student Performance Overview")

    labels = ['Attendance', 'Marks', 'Assignment', 'Participation']
    values = [attendance, marks, assignment, participation * 10]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Performance Indicators")
    st.pyplot(fig)

    # -----------------------------
    # TABLE
    # -----------------------------
    data = {
        "Metric": ["Attendance", "Marks", "Assignment", "Participation"],
        "Value": [attendance, marks, assignment, participation]
    }

    df = pd.DataFrame(data)
    st.table(df)

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("💡 Suggestions")

    if result == "High":
        st.write("- Improve attendance regularly")
        st.write("- Focus on academic performance")
        st.write("- Seek mentoring and guidance")
        st.write("- Increase participation in class activities")

    elif result == "Medium":
        st.write("- Maintain consistency in studies")
        st.write("- Improve weaker subjects")
        st.write("- Participate more actively")

    else:
        st.write("- Keep up the good performance")
        st.write("- Continue active participation")

# -----------------------------
# APPLICATION
# -----------------------------
st.subheader("🌍 Application")

st.write("This system helps institutions identify at-risk students early and take preventive actions.")

st.markdown("---")
st.write("⚠️ Disclaimer: This is an early-warning prototype and may not cover all real-life factors.")