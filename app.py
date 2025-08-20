# app.py — FluxTwin Enterprise (Devices + Smart Advisor + Live Alerts + AI + Holt-Winters + Pro PDF)
from __future__ import annotations
import json
from datetime import datetime, timedelta
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
importlib.reload(pdf_report)

# ---------- New Smart Advisor Utils ----------
from utils.device_catalog import CATALOG
from utils.device_advisor import (
    normalize_inventory, catalog_defaults, detect_upcoming_peak, make_device_actions
)
from utils.alerts import smtp_ready, send_email_alert

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FluxTwin — Enterprise Energy Intelligence", layout="wide")

# ---------- THEME ----------
def themed_css(mode: str = "Dark") -> str:
    dark = """
    :root { --ft-bg:#0b1220; --ft-card:#10182a; --ft-accent:#4ea1ff; --ft-text:#ecf2ff;
            --ft-muted:#94a3b8; --ft-border:rgba(255,255,255,0.08); }"""
    light = """
    :root { --ft-bg:#f8fafc; --ft-card:#ffffff; --ft-accent:#2563eb; --ft-text:#0f172a;
            --ft-muted:#64748b; --ft-border:rgba(15,23,42,0.08); }"""
    base = dark if mode == "Dark" else light
    return f"""
    <style>
    {base}
    html, body, .stApp {{ background: var(--ft-bg) !important; color: var(--ft-text) !important; }}
    h1, h2, h3, h4 {{ color: var(--ft-text) !important; }}
    .ft-card {{
      background: var(--ft-card); border:1px solid var(--ft-border);
      border-radius:18px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.10);
    }}
    .ft-kpi {{ font-size:13px; color:var(--ft-muted); margin-bottom:6px; }}
    .ft-kpi-val {{ font-size:26px; font-weight:800; color:var(--ft-text); }}
    .ft-pill {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid var(--ft-border);
               color:var(--ft-muted); font-size:12px; }}
    .block-container {{ padding-top: 1.2rem; }}
    </style>
    """

# ---------- SIDEBAR ----------
st.sidebar.title("FluxTwin — Controls")
theme_mode = st.sidebar.radio("Theme", ["Dark", "Light"], index=0, horizontal=True)
st.markdown(themed_css(theme_mode), unsafe_allow_html=True)

project_name = st.sidebar.text_input("Project name", value="FluxTwin")
price = st.sidebar.number_input("Electricity price (€/kWh)", min_value=0.0, value=0.25, step=0.01)
mode = st.sidebar.selectbox("Data mode", ["Upload CSV", "Live simulation (in-app)", "Watch realtime CSV (local)"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)

st.sidebar.markdown("---")
usage_type = st.sidebar.selectbox("Usage type", ["Household","Office","Hotel","Factory"], index=1)
has_pv = st.sidebar.checkbox("Has PV system", value=True)
ai_enabled = st.sidebar.toggle("AI Advisor (OpenAI)", value=True)
st.sidebar.caption("If AI=ON, set OPENAI_API_KEY in Streamlit Secrets.")

# ---------- DEVICE INVENTORY (NEW) ----------
st.sidebar.markdown("### Device inventory")
# keep inventory in session
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame(columns=["device","quantity","avg_kw","controllable"])

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Load defaults"):
    st.session_state.inventory_df = catalog_defaults(usage_type)
if col_btn2.button("Clear"):
    st.session_state.inventory_df = pd.DataFrame(columns=["device","quantity","avg_kw","controllable"])

inv_edit = st.sidebar.data_editor(
    st.session_state.inventory_df,
    num_rows="dynamic",
    use_container_width=True,
    key="inv_editor"
)
st.session_state.inventory_df = normalize_inventory(inv_edit)

# ---------- BRAND BAR ----------
logo_path = Path("assets/logo.png")
with st.container():
    c1, c2, c3 = st.columns([0.08, 0.62, 0.30])
    with c1:
        if logo_path.exists():
            st.image(str(logo_path), use_column_width=True)
        else:
            st.markdown('<div class="ft-pill">FLUX</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div>
          <h2 style="margin-bottom:0;">FluxTwin — Enterprise Energy Intelligence</h2>
          <div style="color: var(--ft-muted);">Operational visibility • Forecasted cost • AI actions • Device-level control tips</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.caption(f"PDF module: {getattr(pdf_report, '__version__', 'unknown')}")

# ---------- HELPERS (DATA) ----------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp","consumption_kwh","production_kwh"])
    x = df.copy()
    x.columns = [c.strip().lower() for c in x.columns]
    rename_map = {}
    if "consumption" in x.columns and "consumption_kwh" not in x.columns: rename_map["consumption"] = "consumption_kwh"
    if "kwh" in x.columns and "consumption_kwh" not in x.columns: rename_map["kwh"] = "consumption_kwh"
    if "time" in x.columns and "timestamp" not in x.columns: rename_map["time"] = "timestamp"
    if "datetime" in x.columns and "timestamp" not in x.columns: rename_map["datetime"] = "timestamp"
    if "production" in x.columns and "production_kwh" not in x.columns: rename_map["production"] = "production_kwh"
    if rename_map: x = x.rename(columns=rename_map)
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
        x = x.dropna(subset=["timestamp"]).sort_values("timestamp")
    else:
        x["timestamp"] = pd.date_range(end=datetime.now(), periods=len(x), freq="H")
    for col in ["consumption_kwh","production_kwh"]:
        if col in x.columns: x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    if "consumption_kwh" not in x.columns: x["consumption_kwh"] = 0.0
    if "production_kwh" not in x.columns: x["production_kwh"] = 0.0
    base = ["timestamp","consumption_kwh","production_kwh"]
    extra = [c for c in x.columns if c not in base]
    return x[base + extra]

def load_csv_any(file_or_path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_or_path)
        return standardize_columns(df)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=["timestamp","consumption_kwh","production_kwh"])

def daily_series(df: pd.DataFrame) -> pd.Series:
    x = df.copy(); x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    x = x.dropna(subset=["timestamp"])
    if x.empty: return pd.Series(dtype=float)
    return x.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()

# ---------- FORECAST ----------
def forecast_daily(df: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    s = daily_series(df)
    start_anchor = (pd.to_datetime(df["timestamp"]).max() if not df.empty else datetime.now()) + timedelta(days=1)
    start_date = start_anchor.date()
    if len(s) < 10:
        mean = float(s.mean()) if len(s) else 0.0
        idx = pd.date_range(start=start_date, periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-mean" if len(s) else "naive-mean(empty)"})
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(s, trend="add", seasonal=None).fit()
        f = model.forecast(horizon_days)
        return pd.DataFrame({"date": f.index, "forecast_kwh": f.values, "method": "holt-winters"})
    except Exception:
        mean = float(s.mean())
        idx = pd.date_range(start=start_date, periods=horizon_days, freq="D")
        return pd.DataFrame({"date": idx, "forecast_kwh": [mean]*horizon_days, "method": "naive-fallback"})

# ---------- RULE-BASED & AI ADVISOR ----------
def rule_based_advice(profile: dict, forecast_df: pd.DataFrame | None) -> dict:
    usage = (profile.get("type") or "office").lower()
    has_pv = bool(profile.get("has_pv", True))
    base = {"household": (0.07, 0.12), "office": (0.10, 0.18), "hotel": (0.08, 0.15), "factory": (0.05, 0.12)}
    lo, hi = base.get(usage, (0.08, 0.14))
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df:
        vol = float(np.std(forecast_df["forecast_kwh"])) if len(forecast_df) > 1 else 0.0
        bump = min(0.03, vol / 500.0); hi = min(hi + bump, 0.22)
    if has_pv: lo += 0.01; hi += 0.01
    expected = round((lo + hi) / 2, 3)
    tips = [
        "Create a weekly energy checklist; assign owners.",
        "Enable alerts when hourly usage >120% of baseline.",
    ]
    return {"tips": tips, "expected_savings_pct": expected}

def ai_advice_openai(profile: dict, kpis: dict, fc_df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini") -> dict:
    if OpenAI is None or not api_key:
        return rule_based_advice(profile, fc_df)
    client = OpenAI(api_key=api_key)
    payload = {
        "profile": profile,
        "kpis": kpis,
        "forecast": fc_df[["date","forecast_kwh"]].assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
    }
    sys = ("You are an enterprise energy-efficiency expert. "
           "Return a compact JSON with keys: 'tips' (list) and 'expected_savings_pct' (0..0.30).")
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(payload)}],
        )
        data = json.loads(resp.choices[0].message.content)
        tips = data.get("tips") or []
        pct = float(np.clip(float(data.get("expected_savings_pct", 0.12)), 0.02, 0.30))
        return {"tips": tips, "expected_savings_pct": pct}
    except Exception:
        return rule_based_advice(profile, fc_df)

# ---------- DATA SOURCE ----------
data = pd.DataFrame(columns=["timestamp","consumption_kwh","production_kwh"])
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV (timestamp, consumption_kwh[, production_kwh])", type=["csv"])
    if uploaded: data = load_csv_any(uploaded)
elif mode == "Live simulation (in-app)":
    st.write("Click **Add tick** to append a new reading.")
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = pd.DataFrame(columns=["timestamp","consumption_kwh","production_kwh"])
    if st.button("Add tick (new reading)"):
        now = datetime.now()
        cons = round(np.random.uniform(1.5, 6.0), 2)
        prod = round(np.random.uniform(0.0, 4.0), 2)
        new_row = pd.DataFrame([[now,cons,prod]], columns=["timestamp","consumption_kwh","production_kwh"])
        st.session_state.sim_data = pd.concat([st.session_state.sim_data, new_row], ignore_index=True)
    data = standardize_columns(st.session_state.sim_data)
elif mode == "Watch realtime CSV (local)":
    path = st.text_input("Path to realtime CSV (e.g. realtime_data.csv)", value="realtime_data.csv")
    if st.button("Refresh now"):
        data = load_csv_any(path)

# ---------- GUARD ----------
if data.empty or "consumption_kwh" not in data.columns:
    st.warning("No data yet. Upload a CSV or generate Live simulation ticks.")
    st.stop()
data = standardize_columns(data)

# ---------- EXECUTIVE SUMMARY ----------
total_now = float(data["consumption_kwh"].sum())
avg_now = float(data["consumption_kwh"].mean()) if len(data) else 0.0
max_now = float(data["consumption_kwh"].max()) if len(data) else 0.0
est_cost_now = total_now * price

fc_df = forecast_daily(data, horizon_days=horizon)
fc_total_kwh = float(fc_df["forecast_kwh"].sum()) if not fc_df.empty else 0.0
cost_no_action = fc_total_kwh * price
method_name = fc_df["method"].iloc[0] if ("method" in fc_df.columns and len(fc_df)>0) else "naive-mean"

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

tips_ai = advisor_out["tips"]
savings_pct = float(advisor_out["expected_savings_pct"])
cost_after = cost_no_action * (1.0 - savings_pct)
savings_eur = cost_no_action - cost_after

st.markdown("## Executive summary")
cA, cB, cC, cD = st.columns(4, gap="large")
with cA: st.markdown(f'<div class="ft-card"><div class="ft-kpi">Current period — cost</div><div class="ft-kpi-val">{est_cost_now:,.2f} €</div></div>', unsafe_allow_html=True)
with cB: st.markdown(f'<div class="ft-card"><div class="ft-kpi">Forecast — cost (no action)</div><div class="ft-kpi-val">{cost_no_action:,.2f} €</div></div>', unsafe_allow_html=True)
with cC: st.markdown(f'<div class="ft-card"><div class="ft-kpi">Forecast — cost (after actions)</div><div class="ft-kpi-val">{cost_after:,.2f} €</div></div>', unsafe_allow_html=True)
with cD: st.markdown(f'<div class="ft-card"><div class="ft-kpi">Potential savings</div><div class="ft-kpi-val">{savings_eur:,.2f} €  ({savings_pct*100:.1f}%)</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- 1) Current usage ----------
st.markdown("### 1) Current usage")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Total consumption (kWh)", f"{total_now:,.2f}")
with c2: st.metric("Average sample (kWh)", f"{avg_now:,.2f}")
with c3: st.metric("Max sample (kWh)", f"{max_now:,.2f}")

y_cols = ["consumption_kwh"]
if "production_kwh" in data.columns and data["production_kwh"].any():
    y_cols.append("production_kwh")
fig_hist = px.line(data, x="timestamp", y=y_cols, title="Consumption (and Production) — History",
                   labels={"timestamp":"Time","value":"kWh","variable":"Series"})
st.plotly_chart(fig_hist, use_container_width=True)

# ---------- 2) Forecast — No action ----------
st.markdown("### 2) Forecast — No action")
col1, col2 = st.columns([2,1])
with col1:
    fc_plot = fc_df.rename(columns={"forecast_kwh":"kWh"})
    fc_plot["kind"] = "Forecast"
    hist_for_merge = data[["timestamp","consumption_kwh"]].rename(columns={"timestamp":"date","consumption_kwh":"kWh"})
    hist_for_merge["kind"] = "History"
    merged = pd.concat([hist_for_merge[["date","kWh","kind"]], fc_plot[["date","kWh","kind"]]], ignore_index=True)
    fig_fc = px.line(merged, x="date", y="kWh", color="kind", title=f"History vs Forecast ({method_name})")
    st.plotly_chart(fig_fc, use_container_width=True)
with col2:
    st.markdown(f'<div class="ft-card"><div class="ft-kpi">Forecast total</div><div class="ft-kpi-val">{fc_total_kwh:,.0f} kWh</div>'
                f'<div class="ft-kpi" style="margin-top:8px;">Estimated cost (no action)</div><div class="ft-kpi-val">{cost_no_action:,.2f} €</div></div>', unsafe_allow_html=True)

with st.expander("Daily forecast table (kWh & €) — No action", expanded=False):
    tmp = fc_df.copy()
    tmp["Estimated cost (€)"] = tmp["forecast_kwh"] * price
    tmp = tmp.rename(columns={"date":"Date", "forecast_kwh":"Forecast (kWh)"})
    st.dataframe(tmp, use_container_width=True)

# ---------- 3) Device-level Smart Advisor (NEW) ----------
st.markdown("### 3) Device-level Smart Advisor")
inv_df = st.session_state.inventory_df.copy()
if inv_df.empty:
    st.info("No devices added yet. Use **Load defaults** in the sidebar to add a starting set based on your usage type.")
else:
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

# Detect upcoming peak
peak_info = detect_upcoming_peak(data, fc_df, within_minutes=60)
device_tips = make_device_actions(usage_type, inv_df, peak_info, horizon_days=1)

if peak_info.get("is_peak"):
    when = peak_info["evidence"].get("when", "")
    st.error(f"⚠️ Impending/ongoing peak detected around **{when}**. Threshold-based control is recommended.")
else:
    st.success("No immediate peak detected. Monitoring…")

if device_tips:
    st.markdown("**Actionable control tips (device-specific):**")
    for t in device_tips:
        st.markdown(f"- {t}")
else:
    st.caption("No device-specific actions found (add devices or adjust catalog).")

# ---------- 4) Live Alerts (in-app & optional email) ----------
st.markdown("### 4) Live alerts")
send_emails = st.toggle("Send email alerts (if SMTP secrets exist)", value=False)
if peak_info.get("is_peak"):
    alert_subject = f"[FluxTwin] Peak alert — {project_name}"
    body_lines = [f"Project: {project_name}",
                  f"Detected peak window: {peak_info['evidence']}",
                  "", "Recommended actions:"]
    body_lines += [f"- {t}" for t in device_tips] if device_tips else ["- (no device actions available)"]
    alert_body = "\n".join(body_lines)
    st.code(alert_body, language="text")

    if send_emails:
        if smtp_ready(st.secrets):
            ok, msg = send_email_alert(st.secrets, alert_subject, alert_body, st.secrets.get("ALERT_TO", None))
            if ok:
                st.success("Email alert sent.")
            else:
                st.warning(f"Email not sent: {msg}")
        else:
            st.warning("SMTP secrets missing. Add SMTP_* and ALERT_TO in Streamlit Secrets to enable email alerts.")
else:
    st.caption("Alerts will trigger automatically here if a peak is detected on refresh or new data.")

# ---------- 5) AI Advisor (strategic actions & savings) ----------
st.markdown("### 5) AI Advisor — next 7 days plan")
for t in (tips_ai or []):
    st.markdown(f"- {t}")

# ---------- 6) Forecast — After actions ----------
st.markdown("### 6) Forecast — After actions")
fc_after = fc_df.copy()
fc_after["forecast_kwh_after"] = fc_after["forecast_kwh"] * (1.0 - savings_pct)
fc_after["cost_no"] = fc_after["forecast_kwh"] * price
fc_after["cost_after"] = fc_after["forecast_kwh_after"] * price
cost_after = float(fc_after["cost_after"].sum())
savings_eur = float(fc_after["cost_no"].sum() - fc_after["cost_after"].sum())

colX, colY, colZ = st.columns(3)
with colX: st.metric("Estimated cost (no action)", f"{cost_no_action:,.2f} €")
with colY: st.metric(f"Estimated cost (after actions, -{savings_pct*100:.1f}%)", f"{cost_after:,.2f} €")
with colZ: st.metric("Estimated savings", f"{savings_eur:,.2f} €")

colL, colR = st.columns(2)
with colL:
    dual = pd.DataFrame({
        "date": fc_after["date"],
        "No action (kWh)": fc_after["forecast_kwh"],
        "After actions (kWh)": fc_after["forecast_kwh_after"],
    })
    fig2 = px.line(dual, x="date", y=["No action (kWh)","After actions (kWh)"], title="Forecast (kWh): No action vs After actions")
    st.plotly_chart(fig2, use_container_width=True)
with colR:
    cum = pd.DataFrame({
        "date": fc_after["date"],
        "Cumulative cost — No action (€)": fc_after["cost_no"].cumsum(),
        "Cumulative cost — After actions (€)": fc_after["cost_after"].cumsum(),
    })
    fig3 = px.line(cum, x="date", y=["Cumulative cost — No action (€)","Cumulative cost — After actions (€)"], title="Cumulative cost curves (€)")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption(f"Project: {project_name} • Generated {datetime.now():%Y-%m-%d %H:%M} • PDF: {getattr(pdf_report, '__version__', 'unknown')}")

# ---------- 7) Export (PDF) ----------
st.markdown("### 7) Export")
if st.button("Generate Executive PDF"):
    pdf_path = pdf_report.create_report(
        df=data,
        advisor_text_or_list=(device_tips or tips_ai),
        price_eur_per_kwh=price,
        expected_savings_pct=savings_pct,
        forecast_df=fc_df,
    )
    with open(pdf_path, "rb") as f:
        st.download_button("Download Report (PDF)", f, file_name="FluxTwin_Report.pdf", mime="application/pdf")
