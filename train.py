from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA = Path("data/placement_demo.csv")
MODEL = Path("models/placement_pipeline.joblib")
if not DATA.exists():
    raise SystemExit("Run `python generate_data.py` first.")
df = pd.read_csv(DATA)
X, y = df.drop(columns="placed"), df["placed"]
numeric = X.columns.tolist()
preprocess = ColumnTransformer([("num", Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
]), numeric)])
model = Pipeline([
    ("prep", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                           class_weight="balanced", random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(classification_report(y_test, pred, zero_division=0))
MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL)
print(f"Saved model to {MODEL}")
