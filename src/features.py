import pandas as pd

SERVICE_COLS = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
DROP_COLS = ["customerID", "tenure"] + SERVICE_COLS

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    ).astype(str)
    df["total_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["Average_monthly_spend"] = (df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)).round(2)
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])