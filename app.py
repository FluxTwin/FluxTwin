import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from utils.pdf_report import generate_pdf

# Optional OpenAI (used only if you add a key)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None

load_dotenv()

st.set_page_config(
    page_title="FluxTwin — Energy Analytics MVP",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ FluxTwin — Energy Analytics MVP")
st.caption("Upload data, get analysis, AI summary (optional), 7–30 day forecast, and export a PDF report.")

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.header("Settings")
    price_per_kwh = st.number_input("Electricity price (€ / kWh)", value=0.25, min_value=0.0, step=0.01)
    project_name = st.text_input("Project name", value=os.getenv("FLUXTWIN_APP_NAME", "FluxTwin"))
    forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=7, step=1)
    st.markdown("---")
    st.write("👇 Download a sample dataset to test:")
    try:
        with open("assets/sample_data.csv", "rb") as f:
            st.download_button("Download sample_data.csv", file_name="sample_data.csv", data=f, mime="text/csv")
    except FileNotFoundError:
        st.info("`assets/sample_data.csv` not found in this deployment.")

# ------------------------ HELPERS ------------------------
def get_openai_api_key() -> str:
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key or ""

def safe_read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "timestamp" not in df.columns:
        candidates = [c for c in df.columns if c.lower().strip() in ("time", "date", "datetime")]
        if candidates:
            df = df.rename(columns={candidates[0]: "timestamp"})
    if "consumption_kwh" not in df.columns:
        candidates = [c for c in df.columns if "kwh" in c.lower() or "consum" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "consumption_kwh"})
    if "timestamp" not in df.columns or "consumption_kwh" not in df.columns:
        raise ValueError("CSV must include columns: timestamp, consumption_kwh")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    df = df.dropna(subset=["consumption_kwh"])
    return df

def fit_daily_forecast(daily_series: pd.Series, days_ahead: int = 7):
    """
    Holt–Winters on daily totals when possible; otherwise naive mean.
    Returns: (forecast_series, method_tag)
    """
    if len(daily_series) < 7 or ExponentialSmoothing is None:
        mean_val = float(daily_series.tail(14).mean()) if len(daily_series) >= 1 else 0.0
        idx = pd.date_range(daily_series.index.max() + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        return pd.Series([mean_val]*days_ahead, index=idx), "naive-mean"
    try:
        model = ExponentialSmoothing(daily_series, trend='add', seasonal=None, initialization_method="estimated")
        fitted = model.fit(optimized=True)
        future_index = pd.date_range(daily_series.index.max() + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        forecast = fitted.forecast(days_ahead)
        forecast.index = future_index
        return forecast, "holt-winters"
    except Exception:
        mean_val = float(daily_series.tail(14).mean())
        idx = pd.date_range(daily_series.index.max() + pd.Timedelta(days=1), periods=days_ahead, freq="D")
        return pd.Series([mean_val]*days_ahead, index=idx), "naive-mean"

# ------------------------ MAIN ------------------------
st.subheader("Upload your CSV (columns: timestamp, consumption_kwh)")
uploaded = st.file_uploader("Drag & drop or browse your CSV file", type=["csv"])

if uploaded:
    # Parse
    try:
        df = safe_read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # KPIs (based on uploaded data)
    total_kwh = float(df["consumption_kwh"].sum())
    mean_kwh = float(df["consumption_kwh"].mean())
    max_kwh = float(df["consumption_kwh"].max())
    min_kwh = float(df["consumption_kwh"].min())
    total_cost = float(total_kwh * price_per_kwh)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total consumption", f"{total_kwh:,.2f} kWh")
    c2.metric("Average hourly", f"{mean_kwh:,.2f} kWh")
    c3.metric("Max hourly", f"{max_kwh:,.2f} kWh")
    c4.metric("Estimated cost", f"{total_cost:,.2f} €")

    # Raw consumption chart (history only)
    st.subheader("Consumption chart")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["consumption_kwh"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    st.pyplot(fig)

    # Daily aggregation
    st.subheader("Daily summary")
    daily = (df.set_index("timestamp").resample("D")["consumption_kwh"].sum()).dropna()
    daily_df = daily.reset_index().rename(columns={"consumption_kwh": "daily_kwh"})
    st.dataframe(daily_df)

    # -------- Forecast --------
    st.subheader("Forecast (daily kWh)")
    forecast_series, model_used = fit_daily_forecast(daily, days_ahead=forecast_days)
    st.caption(f"Method: {'Holt-Winters' if model_used=='holt-winters' else 'Naive mean of recent values'}")

    # Forecast chart (history + forecast)
    fig_f, ax_f = plt.subplots()
    ax_f.plot(daily.index, daily.values, label="History")
    ax_f.plot(forecast_series.index, forecast_series.values, label="Forecast")
    ax_f.set_xlabel("Date")
    ax_f.set_ylabel("Daily consumption (kWh)")
    ax_f.legend()
    st.pyplot(fig_f)

    # >>> NEW: forecast totals & cost <<<
    forecast_total_kwh = float(forecast_series.sum())
    forecast_cost = float(forecast_total_kwh * price_per_kwh)
    c5, c6 = st.columns(2)
    c5.metric("Forecast total consumption", f"{forecast_total_kwh:,.2f} kWh")
    c6.metric("Estimated cost (forecast)", f"{forecast_cost:,.2f} €")

    # -------- AI summary (optional) --------
    st.subheader("AI summary")
    api_key = get_openai_api_key()
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            prompt = (
                "Write an executive energy report in English for a Cyprus-based business.\n"
                f"Data: total_kWh={total_kwh:.2f}, avg_hourly={mean_kwh:.2f}, "
                f"max_hourly={max_kwh:.2f}, min_hourly={min_kwh:.2f}, est_cost_eur={total_cost:.2f}.\n"
                f"Next {len(forecast_series)}-day forecast total_kWh={forecast_total_kwh:.2f}, "
                f"forecast_cost_eur={forecast_cost:.2f}.\n"
                "Provide 4–6 actionable, domain-specific recommendations (load shifting, HVAC, lighting, scheduling, PV/battery ROI). "
                "Keep it concise, use bullet points, avoid fluff."
            )
            resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
            ai_text = (resp.output_text or "").strip() or "AI summary unavailable."
        except Exception as e:
            ai_text = (
                "⚠️ AI summary unavailable right now. Showing demo text.\n"
                f"• Total consumption {total_kwh:,.2f} kWh; average hourly {mean_kwh:,.2f} kWh.\n"
                f"• Estimated period cost: {total_cost:,.2f} €.\n"
                f"• {len(forecast_series)}-day forecast total: {forecast_total_kwh:,.2f} kWh "
                f"(~{forecast_cost:,.2f} € at current price).\n"
                "• Consider shifting loads off-peak and optimizing HVAC/lighting during 09:00–18:00."
            )
            st.warning(f"AI call failed: {e}")
    else:
        ai_text = (
            "🎯 Demo summary (no API key detected):\n"
            f"• Total consumption {total_kwh:,.2f} kWh; average hourly {mean_kwh:,.2f} kWh.\n"
            f"• Estimated period cost: {total_cost:,.2f} €.\n"
            f"• {len(forecast_series)}-day forecast total: {forecast_total_kwh:,.2f} kWh "
            f"(~{forecast_cost:,.2f} € at current price).\n"
            "• Consider shifting loads off-peak and optimizing HVAC/lighting during 09:00–18:00."
        )
    st.text_area("AI summary text", ai_text, height=180)

    # -------- Export PDF --------
    st.subheader("Export report")
    if st.button("Generate PDF"):
        # Save consumption chart
        buf_hist = io.BytesIO()
        fig2, ax2 = plt.subplots()
        ax2.plot(df["timestamp"], df["consumption_kwh"])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Consumption (kWh)")
        fig2.savefig(buf_hist, format="png", bbox_inches="tight")
        hist_png = buf_hist.getvalue()

        # Save forecast chart
        buf_fore = io.BytesIO()
        fig3, ax3 = plt.subplots()
        ax3.plot(daily.index, daily.values, label="History")
        ax3.plot(forecast_series.index, forecast_series.values, label="Forecast")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Daily consumption (kWh)")
        ax3.legend()
        fig3.savefig(buf_fore, format="png", bbox_inches="tight")
        fore_png = buf_fore.getvalue()

        stats = {
            "total_kwh": total_kwh,
            "mean_kwh": mean_kwh,
            "max_kwh": max_kwh,
            "min_kwh": min_kwh,
            "total_cost_eur": total_cost,
            # (προαιρετικά: δεν τυπώνονται στον τωρινό PDF πίνακα,
            # αλλά τα κρατάμε αν θέλουμε να τα προσθέσουμε αργότερα)
            "forecast_total_kwh": forecast_total_kwh,
            "forecast_cost_eur": forecast_cost,
        }
        daily_rows = [[str(idx.date()), f"{float(val):,.2f}"] for idx, val in daily.items()]
        forecast_rows = [[str(idx.date()), f"{float(val):,.2f}"] for idx, val in forecast_series.items()]

        tmp_path = "FluxTwin_Report.pdf"
        generate_pdf(
            tmp_path,
            project_name,
            stats,
            ai_text,
            chart_png_bytes=hist_png,
            daily_rows=daily_rows,
            logo_png_path=None,
            forecast_png_bytes=fore_png,
            forecast_rows=forecast_rows,
        )
        with open(tmp_path, "rb") as f:
            st.download_button("Download PDF report", f, file_name="FluxTwin_Report.pdf", mime="application/pdf")
else:
    st.info("Upload the sample_data.csv or your own file to start the analysis.")

