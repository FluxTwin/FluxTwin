import os
import io
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from utils.pdf_report import generate_pdf

# Optional OpenAI import (only used if API key is provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# -------------------------------------------
# Streamlit page config
# -------------------------------------------
st.set_page_config(
    page_title="FluxTwin — Energy Analytics MVP",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ FluxTwin — Energy Analytics MVP")
st.caption("First release: upload data, get analysis, AI summary, and export a PDF report.")

# -------------------------------------------
# Sidebar
# -------------------------------------------
with st.sidebar:
    st.header("Settings")
    price_per_kwh = st.number_input(
        "Electricity price (€ / kWh)", value=0.25, min_value=0.0, step=0.01
    )
    project_name = st.text_input(
        "Project name", value=os.getenv("FLUXTWIN_APP_NAME", "FluxTwin")
    )
    st.markdown("---")
    st.write("👇 Download a sample dataset to test:")
    try:
        with open("assets/sample_data.csv", "rb") as f:
            st.download_button(
                "Download sample_data.csv",
                file_name="sample_data.csv",
                data=f,
                mime="text/csv",
            )
    except FileNotFoundError:
        st.info("`assets/sample_data.csv` not found in this deployment.")

st.subheader("Upload your CSV (columns: timestamp, consumption_kwh)")
uploaded = st.file_uploader("Drag & drop or browse your CSV file", type=["csv"])

# -------------------------------------------
# Helpers
# -------------------------------------------
def get_openai_api_key() -> str:
    # Prefer Streamlit Secrets; fallback to environment variables
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return key or ""

def safe_read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Try to parse a timestamp column with common names
    if "timestamp" not in df.columns:
        # attempt to find a similar column name
        candidates = [c for c in df.columns if c.lower().strip() in ("time", "date", "datetime")]
        if candidates:
            df = df.rename(columns={candidates[0]: "timestamp"})
    if "consumption_kwh" not in df.columns:
        # attempt to infer a consumption column name
        candidates = [c for c in df.columns if "kwh" in c.lower() or "consum" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "consumption_kwh"})
    # Final checks
    if "timestamp" not in df.columns or "consumption_kwh" not in df.columns:
        raise ValueError("CSV must include columns: timestamp, consumption_kwh")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["consumption_kwh"] = pd.to_numeric(df["consumption_kwh"], errors="coerce")
    df = df.dropna(subset=["consumption_kwh"])
    return df

# -------------------------------------------
# Main flow
# -------------------------------------------
if uploaded:
    try:
        df = safe_read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # KPIs
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

    # Chart
    st.subheader("Consumption chart")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["consumption_kwh"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    st.pyplot(fig)

    # Daily aggregation
    st.subheader("Daily summary")
    daily = (
        df.set_index("timestamp")
        .resample("D")["consumption_kwh"]
        .sum()
        .reset_index()
    )
    st.dataframe(daily)

    # AI summary
    st.subheader("AI summary")
    api_key = get_openai_api_key()
    ai_text: str
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            prompt = (
                "Write an executive energy report in English for a Cyprus-based business.\n"
                f"Data: total_kWh={total_kwh:.2f}, avg_hourly={mean_kwh:.2f}, "
                f"max_hourly={max_kwh:.2f}, min_hourly={min_kwh:.2f}, est_cost_eur={total_cost:.2f}.\n"
                "Provide 4–6 actionable, domain-specific recommendations (load shifting, HVAC, lighting, scheduling, PV/battery ROI). "
                "Keep it concise, use bullet points, avoid fluff."
            )
            resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
            ai_text = (resp.output_text or "").strip()
            if not ai_text:
                raise ValueError("Empty AI response")
        except Exception as e:
            ai_text = (
                "⚠️ AI summary unavailable right now. Showing demo text.\n"
                f"• Total consumption {total_kwh:,.2f} kWh; average hourly {mean_kwh:,.2f} kWh.\n"
                f"• Estimated period cost: {total_cost:,.2f} €.\n"
                "• Consider shifting loads off-peak and optimizing HVAC/lighting during 09:00–18:00."
            )
            st.warning(f"AI call failed: {e}")
    else:
        ai_text = (
            "🎯 Demo summary (no API key detected):\n"
            f"• Total consumption {total_kwh:,.2f} kWh; average hourly {mean_kwh:,.2f} kWh.\n"
            f"• Estimated period cost: {total_cost:,.2f} €.\n"
            "• Consider shifting loads off-peak and optimizing HVAC/lighting during 09:00–18:00."
        )
    st.text_area("AI summary text", ai_text, height=180)

    # Export PDF
    st.subheader("Export report")
    if st.button("Generate PDF"):
        # Save the current chart to PNG bytes
        buf = io.BytesIO()
        fig2, ax2 = plt.subplots()
        ax2.plot(df["timestamp"], df["consumption_kwh"])
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Consumption (kWh)")
        fig2.savefig(buf, format="png", bbox_inches="tight")
        chart_png = buf.getvalue()

        stats = {
            "total_kwh": total_kwh,
            "mean_kwh": mean_kwh,
            "max_kwh": max_kwh,
            "min_kwh": min_kwh,
            "total_cost_eur": total_cost,
        }
        daily_rows = [
            [str(r["timestamp"].date()), f"{float(r['consumption_kwh']):,.2f}"]
            for _, r in daily.iterrows()
        ]

        tmp_path = "FluxTwin_Report.pdf"
        # You can pass a logo PNG later via logo_png_path="assets/logo.png"
        generate_pdf(
            tmp_path,
            project_name,
            stats,
            ai_text,
            chart_png_bytes=chart_png,
            daily_rows=daily_rows,
            logo_png_path=None,
        )
        with open(tmp_path, "rb") as f:
            st.download_button(
                "Download PDF report",
                f,
                file_name="FluxTwin_Report.pdf",
                mime="application/pdf",
            )
else:
    st.info("Upload the sample_data.csv or your own file to start the analysis.")

