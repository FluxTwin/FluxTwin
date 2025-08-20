# utils/device_advisor.py
from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .device_catalog import CATALOG

def normalize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["device", "quantity", "avg_kw", "controllable"])
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    if "device" not in x.columns: x["device"] = ""
    if "quantity" not in x.columns: x["quantity"] = 1
    if "avg_kw" not in x.columns: x["avg_kw"] = 0.0
    if "controllable" not in x.columns: x["controllable"] = True
    x["quantity"] = pd.to_numeric(x["quantity"], errors="coerce").fillna(1).astype(int).clip(lower=0)
    x["avg_kw"] = pd.to_numeric(x["avg_kw"], errors="coerce").fillna(0.0)
    x["controllable"] = x["controllable"].astype(bool)
    return x[["device","quantity","avg_kw","controllable"]]

def catalog_defaults(industry: str) -> pd.DataFrame:
    items = CATALOG.get(industry, {})
    rows = []
    for name, meta in items.items():
        rows.append({
            "device": name,
            "quantity": 1,
            "avg_kw": float(meta.get("avg_kw", 0.0)),
            "controllable": bool(meta.get("controllable", True)),
        })
    return pd.DataFrame(rows)

def detect_upcoming_peak(history_df: pd.DataFrame, forecast_df: pd.DataFrame, within_minutes: int = 60) -> dict:
    """
    Heuristic: define dynamic peak as max( 90th percentile of recent hourlies , mean+1.2*std ).
    If the next point in forecast (or the latest reading) exceeds threshold, raise.
    """
    if history_df is None or history_df.empty:
        return {"is_peak": False, "threshold": 0.0, "evidence": None}

    h = history_df.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], errors="coerce")
    h = h.dropna(subset=["timestamp"])
    if h.empty:
        return {"is_peak": False, "threshold": 0.0, "evidence": None}

    # Build hourly series
    s = h.set_index("timestamp")["consumption_kwh"].resample("H").sum().dropna()
    if len(s) < 8:
        thr = float(max(s.max(), s.mean()))
    else:
        thr = float(max(np.percentile(s.values, 90), s.mean() + 1.2*s.std()))

    evidence = {}
    # Check next forecast window
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        f1 = forecast_df.iloc[0]
        if float(f1["forecast_kwh"]) >= thr:
            evidence = {"when": str(f1["date"]), "forecast_kwh": float(f1["forecast_kwh"])}
            return {"is_peak": True, "threshold": thr, "evidence": evidence}

    # Otherwise, check latest reading vs threshold
    last = h.iloc[-1]
    if float(last["consumption_kwh"]) >= thr:
        evidence = {"when": str(last["timestamp"]), "consumption_kwh": float(last["consumption_kwh"])}
        return {"is_peak": True, "threshold": thr, "evidence": evidence}

    return {"is_peak": False, "threshold": thr, "evidence": None}

def make_device_actions(industry: str, inventory_df: pd.DataFrame, peak_info: dict, horizon_days: int = 1) -> list[str]:
    """
    Compose specific, device-level actions around a detected/impending peak.
    """
    tips: list[str] = []
    if inventory_df is None or inventory_df.empty:
        return tips
    inv = normalize_inventory(inventory_df)
    library = CATALOG.get(industry, {})

    when_txt = ""
    if peak_info and peak_info.get("is_peak") and peak_info.get("evidence"):
        when_txt = peak_info["evidence"].get("when", "")

    for _, row in inv.iterrows():
        name = row["device"]
        qty = int(row["quantity"])
        meta = library.get(name, {})
        if not meta:
            continue
        if not bool(meta.get("controllable", True)):
            continue
        actions = meta.get("actions", [])
        if not actions:
            continue

        # Pick top-2 concrete actions and multiply by quantity if meaningful
        base = actions[:2]
        if qty > 1:
            label = f"{name} x{qty}"
        else:
            label = name

        # Frame around WHEN the peak kicks
        prefix = f"[{label}] "
        if when_txt:
            prefix = f"[{when_txt}] " + prefix

        for a in base:
            tips.append(prefix + a)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in tips:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:15]
