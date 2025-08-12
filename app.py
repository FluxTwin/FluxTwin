# app.py — FluxTwin Live Energy Advisor (stable)
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import advisor, pdf_report

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FluxTwin - Live Energy Advisor", layout="wide")

# ---------- HELPERS ----------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and ensure required columns exist."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

    # lowercase / strip spaces
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # try to map common alternatives to expected names
    rename_map = {}
    if "consumption" in df.columns and "consumption_kwh" not in df.columns:
        rename_map["consumption"] = "consumption_kwh"
    if "kwh" in df.columns and "consumption_kwh" not in df.columns:
        rename_map["kwh"] = "consumption_kwh"
    if "time" in df.columns and "timestamp" not in df.columns:
        rename_map["time"] = "timestamp"
    if "datetime" in df.columns and "timestamp" not in df.columns:
        rename_map["datetime"] = "timestamp"
    if "production" in df.columns and "production_kwh" not in df.columns:
        rename_map["production"] = "production_kwh"

    if rename_map:
        df = df.rename(columns=rename_map)

    # timestamp handling
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
    else:
        # fabricate timestamp if missing
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="H")

    # numeric handling
    for col in ["consumption_kwh", "production_kwh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ensure required cols exist
    if "consumption_kwh" not in df.columns:
        df["consumption_kwh"] = 0.0
    if "production_kwh" not in df.columns:
        df["production_kwh"] = 0.0

    # keep only what we need (plus anything else for safety)
    base_cols = ["timestamp", "consumption_kwh", "production_kwh"]
    extra = [c for c in df.columns if c not in base_cols]
    return df[base_cols + extra]


def load_csv_any(file_or_path) -> pd.DataFrame:
    """Read CSV from upload or path; return standardized dataframe."""
    try:
        df = pd.read_csv(file_or_path)
        return standardize_columns(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])


def kpi_row(df: pd.DataFrame, price_per_kwh: float | None = None):
    """Show summary KPIs."""
    total = float(df["consumption_kwh"].sum())
    avg = float(df["consumption_kwh"].mean()) if len(df) else 0.0
    mx = float(df["consumption_kwh"].max()) if len(df) else 0.0
    cols = st.columns(4)
    cols[0].metric("Total consumption", f"{total:,.2f} kWh")
    cols[1].metric("Average sample", f"{avg:,.2f} kWh")
    cols[2].metric("Max sample", f"{mx:,.2f} kWh")
    if price_per_kwh and price_per_kwh > 0:
        est_cost = total * price_per_kwh
        cols[3].metric("Estimated cost", f"{est_cost:,.2f} €")
    else:
        cols[3].metric("Estimated cost", "—")


# ---------- SIDEBAR ----------
st.sidebar.title("Settings")
price = st.sidebar.number_input("Electricity price (€/kWh)", min_value=0.0, value=0.25, step=0.01)
project_name = st.sidebar.text_input("Project name", value="FluxTwin")
mode = st.sidebar.selectbox(
    "Data mode",
    ["Upload CSV", "Live simulation (in-app)", "Watch realtime CSV (local)"]
)

# ---------- DATA SOURCE ----------
data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV (columns: timestamp, consumption_kwh[, production_kwh])", type=["csv"])
    if uploaded:
        data = load_csv_any(uploaded)

elif mode == "Live simulation (in-app)":
    st.write("Click **Add tick** to append a new reading.")
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

    if st.button("Add tick (new reading)"):
        now = datetime.now()
        # simple randoms
        cons = round(np.random.uniform(1.5, 6.0), 2)
        prod = round(np.random.uniform(0.0, 4.0), 2)
        new_row = pd.DataFrame([[now, cons, prod]], columns=["timestamp", "consumption_kwh", "production_kwh"])
        st.session_state.sim_data = pd.concat([st.session_state.sim_data, new_row], ignore_index=True)

    data = standardize_columns(st.session_state.sim_data)

elif mode == "Watch realtime CSV (local)":
    path = st.text_input("Path to realtime CSV (e.g. realtime_data.csv)", value="realtime_data.csv")
    if st.button("Refresh now"):
        data = load_csv_any(path)

# ---------- UI ----------
st.title("⚡ FluxTwin — Live Energy Advisor")
st.caption("Upload data or stream it live, get instant advice & export a clean PDF report.")

if data.empty or "consumption_kwh" not in data.columns:
    st.warning("No data to display yet. Upload a file or generate a few ticks in Live simulation.")
    st.stop()

# ensure correct types / order
data = standardize_columns(data)

# ----- Live overview -----
st.subheader("📊 Live Data Overview")

latest = data.iloc[-1]
cons = float(latest["consumption_kwh"])
prod = float(latest["production_kwh"]) if "production_kwh" in data.columns else 0.0
net = round(cons - prod, 2)

if "production_kwh" in data.columns and data["production_kwh"].any():
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Consumption (kWh)", f"{cons:.2f}")
    c2.metric("Current Production (kWh)", f"{prod:.2f}")
    c3.metric("Net Usage (kWh)", f"{net:.2f}")
else:
    c1, c3 = st.columns(2)
    c1.metric("Current Consumption (kWh)", f"{cons:.2f}")
    c3.metric("Net Usage (kWh)", f"{net:.2f}")

# KPIs on whole dataset
with st.expander("Dataset KPIs", expanded=True):
    kpi_row(data, price_per_kwh=price)

# Advisor
suggestion = advisor.get_advice(consumption=cons, production=prod)
st.info(f"💡 Advisor: {suggestion}")

# Plot
y_cols = ["consumption_kwh"]
if "production_kwh" in data.columns and data["production_kwh"].any():
    y_cols.append("production_kwh")

fig = px.line(
    data,
    x="timestamp",
    y=y_cols,
    title="Consumption vs Production",
    labels={"timestamp": "Time", "value": "kWh", "variable": "Series"},
)
st.plotly_chart(fig, use_container_width=True)

# Optional: show simple daily table if there are many rows/hours
with st.expander("Daily summary table"):
    day = data.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    st.dataframe(day.reset_index().rename(columns={"timestamp": "date", "consumption_kwh": "daily_kwh"}))

# Export PDF
st.subheader("Export")
if st.button("Generate PDF Report"):
    pdf_path = pdf_report.create_report(data, suggestion)
    with open(pdf_path, "rb") as f:
        st.download_button("Download Report (PDF)", f, file_name="FluxTwin_Report.pdf", mime="application/pdf")
