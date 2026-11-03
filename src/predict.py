import json
from pathlib import Path
import joblib, pandas as pd

M = Path(__file__).resolve().parent.parent / "models"
_model = joblib.load(M / "late_delivery_logreg.joblib")
_cols = json.load(open(M / "meta.json"))["columns"]

def predict_risk(orders: pd.DataFrame) -> pd.Series:
    """Probability that each order will be late."""
    return pd.Series(_model.predict_proba(orders[_cols])[:, 1], index=orders.index, name="late_risk")

def flag_top(orders: pd.DataFrame, pct: float = 10) -> pd.Series:
    """Flag the riskiest pct% of a batch (capacity-based rule from Step 10)."""
    risk = predict_risk(orders)
    return risk >= risk.quantile(1 - pct / 100)