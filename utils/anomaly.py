import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame, window: int = 24, z_thresh: float = 3.0):
    """Απλό anomaly detection σε ωριαία κατανάλωση με rolling z-score."""
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp")
    s = x["consumption_kwh"].astype(float)
    roll_mean = s.rolling(window, min_periods=window//2).mean()
    roll_std  = s.rolling(window, min_periods=window//2).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std
    x["zscore"] = z
    x["is_anomaly"] = (z.abs() >= z_thresh)
    return x[["timestamp","consumption_kwh","zscore","is_anomaly"]].dropna()

def failure_score(df: pd.DataFrame, days: int = 14):
    """Πολύ απλό health metric: αν η παραγωγή/απόδοση πέφτει σταδιακά."""
    y = df.copy()
    y["timestamp"] = pd.to_datetime(y["timestamp"])
    day = y.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    if len(day) < 7:
        return {"score": 0.0, "note": "Insufficient data"}
    recent = day.tail(days)
    trend = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]  # slope
    # Κανονικοποίηση slope σε [0..1] (όσο πιο μεγάλη αρνητική, τόσο χειρότερα)
    norm = float(np.clip(-trend / (recent.mean()*0.05 + 1e-6), 0, 1))
    return {"score": norm, "note": "Downward trend" if norm > 0.6 else "Stable"}
