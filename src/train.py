from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT /"data"/"processed"/"cleaned_telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

CANDIDATES = {
    "logreg": LogisticRegression(C=5, max_iter=1000, class_weight="balanced", random_state=2),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=2),
}


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def split_xy(df):
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn"])         
    return X, y


def make_split(X, y):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)


def build_pipeline(classifier, X):
    X_feat = add_features(X)
    num = X_feat.select_dtypes(include="number").columns.tolist()
    cat = X_feat.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
    return Pipeline([("features", FunctionTransformer(add_features)),
                     ("preprocessor", pre),
                     ("classifier", classifier)])


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    return {"roc_auc": roc_auc_score(y_test, proba),
            "recall": recall_score(y_test, pipe.predict(X_test))}


def main():
    X, y = split_xy(load_data())
    X_train, X_test, y_train, y_test = make_split(X, y)

    best_name, best_pipe, best_auc = None, None, -1
    for name, clf in CANDIDATES.items():
        pipe = build_pipeline(clf, X_train)
        pipe.fit(X_train, y_train)                 
        m = evaluate(pipe, X_test, y_test)
        print(f"{name}: ROC-AUC={m['roc_auc']:.3f}  recall={m['recall']:.3f}")
        if m["roc_auc"] > best_auc:
            best_name, best_pipe, best_auc = name, pipe, m["roc_auc"]

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved best model ({best_name}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()