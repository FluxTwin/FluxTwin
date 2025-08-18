# app.py — FluxTwin Enterprise (Branding + Theme Toggle + AI + Holt-Winters + Pro PDF)
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional OpenAI (AI advisor). If not present or no key → fallback to rules.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Pro PDF module (external) ----------
from utils import pdf_report
import importlib
importlib.reload(pdf_report)  # ensure latest version loads

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FluxTwin — Enterprise Energy Intelligence", layout="wide")

# ---------- THEME HELPER ----------
def themed_css(mode: str = "Dark") -> str:
    dark = """
    :root {
      --ft-bg: #0b1220;
      --ft-card: #10182a;
      --ft-accent: #4ea1ff;
      --ft-text: #ecf2ff;
      --ft-muted: #94a3b8;
      --ft-border: rgba(255,255,255,0.08);
    }
    """
    light = """
    :root {
      --ft-bg: #f8fafc;
      --ft-card: #ffffff;
      --ft-accent: #2563eb;
      --ft-text: #0f172a;
      --ft-muted: #64748b;
      --ft-border: rgba(15,23,42,0.08);
    }
    """
    base = dark if mode == "Dark" else light
    return f"""
    <style>
    {base}
    html, body, .stApp {{ background: var(--ft-bg) !important; color: var(--ft-text) !important; }}
    h1, h2, h3, h4 {{ color: var(--ft-text) !important; }}
    .ft-card {{
      background: var(--ft-card);
      border: 1px solid var(--ft-border);
      border-radius: 18px;
      padding: 18px 18px 8px 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }}
    .ft-kpi {{ font-size: 14px; color: var(--ft-muted); margin-bottom: 4px; }}
    .ft-kpi-val {{ font-size: 22px; font-weight: 700; color: var(--ft-text); }}
    .ft-brand {{
      display:flex; align-items:center; gap:12px; border-bottom:1px solid var(--ft-border);
      padding:8px 0 16px 0; margin-bottom:12px;
    }}
    .ft-brand img {{ height:32px; width:auto; border-radius:8px; }}
    .ft-brand .title {{ font-weight:800; font-size:20px; letter-spacing:.3px; }}
    .ft-brand .subtitle {{ color: var(--ft-muted); font-size:13px; margin-top:-4px; }}
    .ft-pill {{
      display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--ft-border);
      color: var(--ft-muted); font-size:12px;
    }}
    .block-container {{ padding-top: 1.2rem; }}
    </style>
    """

# ---------- SIDEBAR (with theme) ----------
st.sidebar.title("FluxTwin — Controls")
theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)
st.markdown(themed_css(theme_mode), unsafe_allow_html=True)

project_name = st.sidebar.text_input("Project name", value="FluxTwin")
price = st.sidebar.number_input("Electricity price (€/kWh)", min_value=0.0, value=0.25, step=0.01)
mode = st.sidebar.selectbox("Data mode", ["Upload CSV", "Live simulation (in-app)", "Watch realtime CSV (local)"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 7)

st.sidebar.markdown("---")
usage_type = st.sidebar.selectbox("Usage type", ["Household","Office","Hotel","Factory"], index=1)
has_pv = st.sidebar.checkbox("Has PV system", value=True)
ai_enabled = st.sidebar.toggle("AI Advisor (OpenAI)", value=True)
st.sidebar.caption("If AI=ON, set OPENAI_API_KEY in Streamlit Secrets.")

# ---------- BRAND BAR ----------
logo_path = Path("assets/logo.png")
with st.container():
    col_logo, col_title, col_right = st.columns([0.08, 0.62, 0.30])
    with col_logo:
        if logo_path.exists():
            st.markdown(f'<div class="ft-brand"><img src="data:image/png;base64,{Path(logo_path).read_bytes().hex()}"/></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ft-brand"><span class="ft-pill">FLUX</span></div>', unsafe_allow_html=True)
    with col_title:
        st.markdown(f"""
        <div class="ft-brand">
          <div>
            <div class="title">FluxTwin — Enterprise Energy Intelligence</div>
            <div class="subtitle">Operational visibility • Forecasted cost • AI actions • Measurable savings</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        st.caption(f"PDF module: {getattr(pdf_report, '__version__', 'unknown')}")

# ---------- DATA HELPERS ----------
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


# ---------- FORECAST (Holt-Winters + fallback) ----------
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

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(s, trend="add", seasonal=None).fit()
        f = model.forecast(horizon_days)
        return pd.DataFrame({"date": f.index, "forecast_kwh": f.values, "method": "holt-winters"})
    except Exception:
        mean = float(s.mean())
        idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-fallback"})


# ---------- RULE-BASED ADVISOR (fallback) ----------
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


# ---------- AI ADVISOR (OpenAI JSON) ----------
def ai_advice_openai(profile: dict, kpis: dict, fc_df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> dict:
    """
    Returns dict: { tips: [..], expected_savings_pct: 0.xx }
    JSON structured output for stability. Fallback to rules on error.
    """
    if OpenAI is None or not api_key:
        return rule_based_advice(profile, fc_df)

    client = OpenAI(api_key=api_key)
    payload = {
        "profile": profile,
        "kpis": kpis,
        "forecast": fc_df[["date", "forecast_kwh"]].assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
    }
    sys = (
        "You are an enterprise energy-efficiency expert. "
        "Return a compact JSON with keys: 'tips' (list of concise, actionable steps) and "
        "'expected_savings_pct' (float 0..0.30 realistic). Tailor to the next 7 days."
    )
    user = "Generate precise actions (off-peak timing, HVAC setpoints, load shifting, PV usage)."
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
        data = json.loads(resp.choices[0].message.content)
        tips = data.get("tips") or []
        pct = float(data.get("expected_savings_pct", 0.12))
        pct = float(np.clip(pct, 0.02, 0.30))
        if len(tips) < 3:
            tips += rule_based_advice(profile, fc_df)["tips"][:3]
        return {"tips": tips, "expected_savings_pct": pct}
    except Exception:
        return rule_based_advice(profile, fc_df)


# ---------- DATA SOURCE ----------
data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (timestamp, consumption_kwh[, production_kwh])", type=["csv"])
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

# ---------- CONTENT ----------
if data.empty or "consumption_kwh" not in data.columns:
    st.warning("No data yet. Upload a CSV or generate Live simulation ticks.")
    st.stop()

data = standardize_columns(data)

# SECTION 1 — CURRENT PERIOD (NOW)
st.markdown("### 1) Current period — where you stand today")
with st.container():
    colA, colB, colC, colD = st.columns(4)
    total_now = float(data["consumption_kwh"].sum())
    avg_now = float(data["consumption_kwh"].mean()) if len(data) else 0.0
    max_now = float(data["consumption_kwh"].max()) if len(data) else 0.0
    est_cost_now = total_now * price

    with colA:
        st.markdown('<div class="ft-card"><div class="ft-kpi">Total consumption</div><div class="ft-kpi-val">'
                    f'{total_now:,.2f} kWh</div></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="ft-card"><div class="ft-kpi">Average sample</div><div class="ft-kpi-val">'
                    f'{avg_now:,.2f} kWh</div></div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="ft-card"><div class="ft-kpi">Max sample</div><div class="ft-kpi-val">'
                    f'{max_now:,.2f} kWh</div></div>', unsafe_allow_html=True)
    with colD:
        st.markdown('<div class="ft-card"><div class="ft-kpi">Estimated cost (current period)</div>'
                    f'<div class="ft-kpi-val">{est_cost_now:,.2f} €</div></div>', unsafe_allow_html=True)

    y_cols = ["consumption_kwh"]
    if "production_kwh" in data.columns and data["production_kwh"].any():
        y_cols.append("production_kwh")
    fig_hist = px.line(
        data, x="timestamp", y=y_cols,
        title="Consumption (and Production) — History",
        labels={"timestamp": "Time", "value": "kWh", "variable": "Series"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# SECTION 2 — FORECAST (NO ACTION)
st.markdown("### 2) Forecast — if you change nothing (No action)")
fc_df = forecast_daily(data, horizon_days=horizon)
fc_total_kwh = float(fc_df["forecast_kwh"].sum()) if not fc_df.empty else 0.0
cost_no_action = fc_total_kwh * price

col1, col2 = st.columns([2,1])
with col1:
    fc_plot = fc_df.rename(columns={"forecast_kwh":"kWh"})
    fc_plot["kind"] = "Forecast"
    hist_for_merge = data[["timestamp","consumption_kwh"]].rename(columns={"timestamp":"date","consumption_kwh":"kWh"})
    hist_for_merge["kind"] = "History"
    merged = pd.concat([hist_for_merge[["date","kWh","kind"]], fc_plot[["date","kWh","kind"]]], ignore_index=True)
    fig_fc = px.line(merged, x="date", y="kWh", color="kind", title=f"History vs Forecast ({fc_df['method'].iloc[0]})")
    st.plotly_chart(fig_fc, use_container_width=True)
with col2:
    st.markdown('<div class="ft-card"><div class="ft-kpi">Forecast total</div>'
                f'<div class="ft-kpi-val">{fc_total_kwh:,.0f} kWh</div>'
                '<div class="ft-kpi" style="margin-top:8px;">Estimated cost (no action)</div>'
                f'<div class="ft-kpi-val">{cost_no_action:,.2f} €</div></div>', unsafe_allow_html=True)

with st.expander("Daily forecast table (kWh & €)"):
    tmp = fc_df.copy()
    tmp["Estimated cost (€)"] = tmp["forecast_kwh"] * price
    tmp = tmp.rename(columns={"date":"Date", "forecast_kwh":"Forecast (kWh)"})
    st.dataframe(tmp, use_container_width=True)

# SECTION 3 — AI ADVISOR (ACTIONS)
st.markdown("### 3) AI Advisor — what to do in the next 7 days")
profile = {"type": usage_type.lower(), "has_pv": has_pv, "price_eur_per_kwh": price}
kpis = {"total_kwh": total_now, "avg_kwh": avg_now, "max_kwh": max_now}
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""

if ai_enabled and api_key:
    st.caption("Powered by OpenAI.")
    advisor_out = ai_advice_openai(profile, kpis, fc_df, api_key)
elif ai_enabled and not api_key:
    st.warning("AI is ON but no OPENAI_API_KEY found in Streamlit Secrets — using rule-based fallback.")
    advisor_out = rule_based_advice(profile, fc_df)
else:
    st.caption("Rule-based plan (enable AI + add OPENAI_API_KEY in Secrets for richer advice).")
    advisor_out = rule_based_advice(profile, fc_df)

tips_list = advisor_out["tips"]
savings_pct = float(advisor_out["expected_savings_pct"])
for t in tips_list:
    st.markdown(f"- {t}")

# SECTION 4 — FORECAST (AFTER ACTIONS)
st.markdown("### 4) Forecast — if you follow the actions (After actions)")
cost_after = cost_no_action * (1.0 - savings_pct)
savings_eur = cost_no_action - cost_after

colA2, colB2, colC2 = st.columns(3)
with colA2:
    st.markdown('<div class="ft-card"><div class="ft-kpi">Estimated cost (no action)</div>'
                f'<div class="ft-kpi-val">{cost_no_action:,.2f} €</div></div>', unsafe_allow_html=True)
with colB2:
    st.markdown(f'<div class="ft-card"><div class="ft-kpi">Estimated cost (after actions, -{savings_pct*100:.1f}%)</div>'
                f'<div class="ft-kpi-val">{cost_after:,.2f} €</div></div>', unsafe_allow_html=True)
with colC2:
    st.markdown('<div class="ft-card"><div class="ft-kpi">Estimated savings</div>'
                f'<div class="ft-kpi-val">{savings_eur:,.2f} €</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Project: {project_name} • Generated {datetime.now():%Y-%m-%d %H:%M} • PDF: {getattr(pdf_report, '__version__', 'unknown')}")

# SECTION 5 — EXPORT (PDF via utils/pdf_report)
st.markdown("### 5) Export")
if st.button("Generate Executive PDF"):
    pdf_path = pdf_report.create_report(
        df=data,
        advisor_text_or_list=tips_list,   # μπορείς να περάσεις και σκέτο string
        price_eur_per_kwh=price,
        expected_savings_pct=savings_pct,
        forecast_df=fc_df,
    )
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Report (PDF)",
            f,
            file_name="FluxTwin_Report.pdf",
            mime="application/pdf",
        )
