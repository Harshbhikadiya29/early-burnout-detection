# project/train_model.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

INP = Path("data/processed/burnout_demo.csv")
OUT = Path("model/burnout_model.pkl")

def main():
    if not INP.exists():
        raise FileNotFoundError(f"Missing processed data: {INP}. Run project/data_clean.py first.")

    df = pd.read_csv(INP)
    X = df[["hours","workload","sleep","stress"]]
    y = df["burnout_label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = LogisticRegression(max_iter=500, class_weight="balanced", C=0.8)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("cal", CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=5)),
    ])

    pipe.fit(X_tr, y_tr)
    print("\n=== Test Report ===")
    print(classification_report(y_te, pipe.predict(X_te)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT.as_posix())
    print(f"\nSaved model → {OUT.resolve()}")

if __name__ == "__main__":
    main()
