# app.py — FluxTwin (AI + Holt-Winters Forecast, cost before/after, single-file)
from __future__ import annotations
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional OpenAI (AI advisor). If not present or no key, we'll fallback to rules.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------- PAGE CONFIG -------------
st.set_page_config(page_title="FluxTwin — Live Energy Advisor", layout="wide")

# ------------- HELPERS (DATA) -------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV columns into [timestamp, consumption_kwh, production_kwh]."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    # Map common alternatives
    rename_map = {}
    if "consumption" in x.columns and "consumption_kwh" not in x.columns:
        rename_map["consumption"] = "consumption_kwh"
    if "kwh" in x.columns and "consumption_kwh" not in x.columns:
        rename_map["kwh"] = "consumption_kwh"
    if "time" in x.columns and "timestamp" not in x.columns:
        rename_map["time"] = "timestamp"
    if "datetime" in x.columns and "timestamp" not in x.columns:
        rename_map["datetime"] = "timestamp"
    if "production" in x.columns and "production_kwh" not in x.columns:
        rename_map["production"] = "production_kwh"
    if rename_map:
        x = x.rename(columns=rename_map)

    # Timestamps
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
        x = x.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        x["timestamp"] = pd.date_range(end=datetime.now(), periods=len(x), freq="H")

    # Numerics
    for col in ["consumption_kwh", "production_kwh"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    if "consumption_kwh" not in x.columns:
        x["consumption_kwh"] = 0.0
    if "production_kwh" not in x.columns:
        x["production_kwh"] = 0.0

    base = ["timestamp", "consumption_kwh", "production_kwh"]
    extra = [c for c in x.columns if c not in base]
    return x[base + extra]


def load_csv_any(file_or_path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_or_path)
        return standardize_columns(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])


def daily_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate to daily kWh on consumption."""
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    s = x.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    return s


# ------------- FORECAST (Holt-Winters + fallback) -------------
def forecast_daily(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """
    Returns dataframe: [date, forecast_kwh, method]
    Uses Holt-Winters (trend add, no seasonality) if enough data, else naive mean.
    """
    s = daily_series(df)
    if len(s) < 10:
        mean = float(s.mean()) if len(s) else 0.0
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-mean"})

    # Try Holt-Winters
    try:
        # Lazy import to be compatible with multiple Python versions on Streamlit Cloud
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(s, trend="add", seasonal=None).fit()
        f = model.forecast(horizon_days)
        return pd.DataFrame({"date": f.index, "forecast_kwh": f.values, "method": "holt-winters"})
    except Exception:
        mean = float(s.mean())
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-fallback"})


# ------------- RULE-BASED ADVISOR (fallback) -------------
def rule_based_advice(profile: dict, forecast_df: pd.DataFrame | None) -> dict:
    usage = (profile.get("type") or "office").lower()
    has_pv = bool(profile.get("has_pv", True))

    base = {
        "household": (0.07, 0.12),
        "office":    (0.10, 0.18),
        "hotel":     (0.08, 0.15),
        "factory":   (0.05, 0.12),
    }
    lo, hi = base.get(usage, (0.08, 0.14))

    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df:
        vol = float(np.std(forecast_df["forecast_kwh"])) if len(forecast_df) > 1 else 0.0
        bump = min(0.03, vol / 500.0)
        hi = min(hi + bump, 0.22)
    if has_pv:
        lo += 0.01; hi += 0.01
    expected = round((lo + hi) / 2, 3)

    tips = []
    if usage in ("office", "hotel"):
        tips += [
            "Shift non-critical loads to off-peak hours (servers backup, laundry).",
            "HVAC setpoints: +1 °C in cooling / −1 °C in heating outside peak hours.",
            "Install occupancy sensors & daylight dimming in corridors/meeting rooms.",
            "Weekly schedule review: turn off AHUs/VRF per zone after 18:00.",
        ]
    if usage == "hotel":
        tips += [
            "Hot-water recirculation: timer + temperature band control.",
            "Pool filtration to off-peak; cover pool at night to reduce losses.",
        ]
    if usage == "factory":
        tips += [
            "Compressors: fix leaks, cascaded pressure setpoints, regular maintenance.",
            "Stagger high-load machines (15-min ramps) to flatten peaks.",
        ]
    if usage == "household":
        tips += [
            "Time-shift dishwasher/washing machine to off-peak.",
            "Smart plugs for standby killers (TV, set-top boxes, chargers).",
        ]
    if has_pv:
        tips += [
            "Run heat-pumps/boilers for pre-heating water during PV peak (11:00–15:00).",
            "Consider small battery (3–5 kWh) to shave evening peaks.",
        ]
    tips += [
        "Create a simple weekly energy checklist; assign an owner per system.",
        "Enable automated alerts when hourly usage >120% of baseline.",
    ]
    return {"tips": tips, "expected_savings_pct": expected}


# ------------- AI ADVISOR (OpenAI JSON) -------------
def ai_advice_openai(profile: dict, kpis: dict, fc_df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """
    Calls OpenAI and returns dict: { tips: [..], expected_savings_pct: 0.xx }
    Uses JSON structured output for stability. Falls back to rules on error.
    """
    if OpenAI is None or not api_key:
        return rule_based_advice(profile, fc_df)

    client = OpenAI(api_key=api_key)
    # Keep context small & structured
    payload = {
        "profile": profile,
        "kpis": kpis,
        "forecast": fc_df[["date", "forecast_kwh"]].assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
    }
    sys = (
        "You are an energy-efficiency expert. "
        "Return a compact JSON with 'tips' (list of concise, actionable steps) and "
        "'expected_savings_pct' (0..0.25 realistic). Target concrete actions for the next 7 days."
    )
    user = (
        "Generate tailored actions to reduce electricity cost based on the provided profile, KPIs and forecast. "
        "Avoid generic statements. Be precise about timing (off-peak), HVAC setpoints, load shifting, and PV usage."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"INPUT:\n{json.dumps(payload)}\n\n{user}"},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        tips = data.get("tips") or []
        pct = float(data.get("expected_savings_pct", 0.12))
        # Clamp to sane range
        pct = float(np.clip(pct, 0.02, 0.30))
        # If tips too few, add a couple of robust ones
        if len(tips) < 3:
            tips += rule_based_advice(profile, fc_df)["tips"][:3]
        return {"tips": tips, "expected_savings_pct": pct}
    except Exception:
        # Any error → safe fallback
        return rule_based_advice(profile, fc_df)


# ------------- UI: SIDEBAR -------------
st.sidebar.title("Settings")
price = st.sidebar.number_input("Electricity price (€/kWh)", min_value=0.0, value=0.25, step=0.01)
project_name = st.sidebar.text_input("Project name", value="FluxTwin")
mode = st.sidebar.selectbox("Data mode", ["Upload CSV", "Live simulation (in-app)", "Watch realtime CSV (local)"])
ai_enabled = st.sidebar.toggle("AI Advisor (OpenAI)", value=True)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 7)
usage_type = st.sidebar.selectbox("Usage type", ["Household", "Office", "Hotel", "Factory"], index=1)
has_pv = st.sidebar.checkbox("Has PV system", value=True)

# ------------- DATA SOURCE -------------
data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (columns: timestamp, consumption_kwh[, production_kwh])", type=["csv"])
    if uploaded:
        data = load_csv_any(uploaded)

elif mode == "Live simulation (in-app)":
    st.write("Click **Add tick** to append a new reading.")
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])
    if st.button("Add tick (new reading)"):
        now = datetime.now()
        cons = round(np.random.uniform(1.5, 6.0), 2)
        prod = round(np.random.uniform(0.0, 4.0), 2)
        new_row = pd.DataFrame([[now, cons, prod]], columns=["timestamp", "consumption_kwh", "production_kwh"])
        st.session_state.sim_data = pd.concat([st.session_state.sim_data, new_row], ignore_index=True)
    data = standardize_columns(st.session_state.sim_data)

elif mode == "Watch realtime CSV (local)":
    path = st.text_input("Path to realtime CSV (e.g. realtime_data.csv)", value="realtime_data.csv")
    if st.button("Refresh now"):
        data = load_csv_any(path)

# ------------- MAIN UI -------------
st.title("⚡ FluxTwin — Live Energy Advisor")
st.caption("Upload data or stream it live. See forecasted cost, AI recommendations, and expected savings.")

if data.empty or "consumption_kwh" not in data.columns:
    st.warning("No data to display yet. Upload a file or generate a few ticks in Live simulation.")
    st.stop()

data = standardize_columns(data)

# KPIs (dataset)
total = float(data["consumption_kwh"].sum())
avg = float(data["consumption_kwh"].mean()) if len(data) else 0.0
mx = float(data["consumption_kwh"].max()) if len(data) else 0.0

# Forecast
fc_df = forecast_daily(data, horizon_days=horizon)
fc_total_kwh = float(fc_df["forecast_kwh"].sum()) if not fc_df.empty else 0.0

# Profile & KPIs object for AI
profile = {"type": usage_type.lower(), "has_pv": has_pv, "price_eur_per_kwh": price}
kpis = {"total_kwh": total, "avg_kwh": avg, "max_kwh": mx}

# Advisor (AI or fallback)
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if ai_enabled and api_key:
    advisor_out = ai_advice_openai(profile, kpis, fc_df, api_key)
elif ai_enabled and not api_key:
    st.warning("AI is ON but no OPENAI_API_KEY found in Streamlit Secrets — using rule-based fallback.")
    advisor_out = rule_based_advice(profile, fc_df)
else:
    advisor_out = rule_based_advice(profile, fc_df)

tips_list = advisor_out["tips"]
savings_pct = float(advisor_out["expected_savings_pct"])

# ---- TOP METRICS: COST BEFORE / AFTER ----
est_cost_no_action = fc_total_kwh * price
est_cost_after = est_cost_no_action * (1.0 - savings_pct)
savings_eur = est_cost_no_action - est_cost_after
cols_top = st.columns(4)
cols_top[0].metric("Estimated cost (no action)", f"{est_cost_no_action:,.2f} €")
cols_top[1].metric("Estimated cost (after actions)", f"{est_cost_after:,.2f} €")
cols_top[2].metric("Savings (€)", f"{savings_eur:,.2f}")
cols_top[3].metric("Savings (%)", f"{savings_pct*100:.1f}%")

# ---- LIVE SNAPSHOT ----
st.subheader("📊 Live snapshot")
latest = data.iloc[-1]
cons = float(latest["consumption_kwh"])
prod = float(latest.get("production_kwh", 0.0))
net = cons - prod
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Consumption", f"{cons:.2f} kWh")
c2.metric("Current Production", f"{prod:.2f} kWh")
c3.metric("Net Usage", f"{net:.2f} kWh")
c4.metric("Dataset total", f"{total:,.2f} kWh")

# ---- CHART: History + Forecast ----
st.subheader("📈 History + Forecast")
hist = data[["timestamp", "consumption_kwh"]].rename(columns={"timestamp": "date", "consumption_kwh": "kWh"})
hist["kind"] = "History"
fc_plot = fc_df.rename(columns={"forecast_kwh": "kWh"})
fc_plot["kind"] = "Forecast"
merged = pd.concat([hist[["date", "kWh", "kind"]], fc_plot[["date", "kWh", "kind"]]], ignore_index=True)
fig = px.line(merged, x="date", y="kWh", color="kind", title="Consumption (History vs Forecast)")
st.plotly_chart(fig, use_container_width=True)

# ---- FORECAST TABLE ----
with st.expander("Daily forecast (kWh & €)"):
    tmp = fc_df.copy()
    tmp["Estimated cost (€)"] = tmp["forecast_kwh"] * price
    tmp = tmp.rename(columns={"date": "Date", "forecast_kwh": "Forecast (kWh)"})
    st.dataframe(tmp, use_container_width=True)

# ---- NEXT 7 DAYS ACTION PLAN (AI / fallback) ----
st.subheader("🧠 Next 7 Days Action Plan")
if ai_enabled and api_key:
    st.caption("Powered by AI (OpenAI).")
else:
    st.caption("Rule-based plan (enable AI + add OPENAI_API_KEY in Secrets for richer advice).")

for t in tips_list:
    st.markdown(f"- {t}")

st.markdown("---")
st.caption(f"Project: {project_name} • Generated {datetime.now():%Y-%m-%d %H:%M}")
