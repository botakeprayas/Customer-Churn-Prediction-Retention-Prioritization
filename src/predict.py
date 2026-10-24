import joblib
import pandas as pd

from src.train import MODEL_PATH, ROOT, load_data, make_split, split_xy

HIGH, MEDIUM = 0.7, 0.4       


def risk_band(p: float) -> str:
    return "High" if p >= HIGH else "Medium" if p >= MEDIUM else "Low"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def score(model, X):
    return model.predict_proba(X.drop(columns=["Churn"], errors="ignore"))[:, 1]


def top_risky(model, df, n=100):
    ids = df["customerID"].values if "customerID" in df.columns else df.index.values
    out = pd.DataFrame({"customer_id": ids, "risk_score": score(model, df)})
    out["risk_band"] = out["risk_score"].map(risk_band)
    return out.sort_values("risk_score", ascending=False).head(n)


if __name__ == "__main__":
    X, y = split_xy(load_data())
    _, X_test, _, _ = make_split(X, y)
    model = load_model()

    print(pd.Series([risk_band(p) for p in score(model, X_test)]).value_counts())

    out_path = ROOT / "data" / "processed" / "top100_risky.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_risky(model, X_test).to_csv(out_path, index=False)
    print("Saved", out_path)