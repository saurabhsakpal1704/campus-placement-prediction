# Campus Placement Prediction

A reproducible machine-learning starter project that trains a classification model to estimate placement outcomes from academic and skill-related features.

> **Demo-data note:** `generate_data.py` creates synthetic records for demonstration. It is not real university placement data and must not be used to make decisions about real students.

## Features
- Reproducible synthetic dataset generator
- Train/test split and preprocessing pipeline
- Random Forest classifier with evaluation metrics
- Streamlit interface for interactive demo predictions
- Feature-importance chart and saved model pipeline

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

## Files
- `generate_data.py` — creates `data/placement_demo.csv`
- `train.py` — trains and evaluates the model; saves `models/placement_pipeline.joblib`
- `app.py` — interactive Streamlit app
- `data/` — generated demo CSV (created when you run generator)
- `models/` — trained model (created by training script)

## Input features
CGPA, internships, projects, coding score, communication score, aptitude score and backlogs. The target is `placed` (0/1).

## Responsible use
The generated dataset is synthetic. Model scores are illustrative only. Real hiring decisions require validated, representative data, bias testing, consent, and human review.
