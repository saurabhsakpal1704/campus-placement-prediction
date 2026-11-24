from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Campus Placement Demo", page_icon="🎓", layout="centered")
st.title("🎓 Campus Placement Prediction")
st.caption("Educational demo using synthetic data — not a real hiring assessment.")
model_path = Path("models/placement_pipeline.joblib")
if not model_path.exists():
    st.warning("Model not found. Run `python generate_data.py` and `python train.py` first.")
    st.stop()
model = joblib.load(model_path)
with st.form("prediction"):
    c1, c2 = st.columns(2)
    cgpa = c1.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
    internships = c2.number_input("Internships", 0, 5, 1)
    projects = c1.number_input("Projects", 0, 10, 3)
    coding = c2.slider("Coding score", 0, 100, 70)
    communication = c1.slider("Communication score", 0, 100, 70)
    aptitude = c2.slider("Aptitude score", 0, 100, 70)
    backlogs = c1.number_input("Backlogs", 0, 5, 0)
    submitted = st.form_submit_button("Estimate demo outcome")
if submitted:
    row = pd.DataFrame([{"cgpa":cgpa,"internships":internships,"projects":projects,
        "coding_score":coding,"communication_score":communication,
        "aptitude_score":aptitude,"backlogs":backlogs}])
    probability = model.predict_proba(row)[0,1]
    st.metric("Model-estimated probability (synthetic demo)", f"{probability:.1%}")
    st.info("This is a model output on synthetic training data, not a prediction of an individual's real placement prospects.")
