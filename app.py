import os, io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.pdf_report import generate_pdf

load_dotenv()
st.set_page_config(page_title="FluxTwin — Energy Analytics MVP", page_icon="⚡", layout="wide")

st.title("⚡ FluxTwin — Energy Analytics MVP")
st.caption("First release: upload data, get analysis, AI summary, and export a PDF report.")

with st.sidebar:
    st.header("Settings")
    price_per_kwh = st.number_input("Electricity price (€ / kWh)", value=0.25, min_value=0.0, step=0.01)
    project_name = st.text_input("Project name", value=os.getenv("FLUXTWIN_APP_NAME", "FluxTwin"))
    st.markdown("---")
    st.write("👇 Download a sample dataset to test:")
    with open("assets/sample_data.csv", "rb") as f:
        st.download_button("Download sample_data.csv", file_name="sample_data.csv", data=f, mime="text/csv")

st.subheader("Upload your CSV (columns: timestamp, consumption_kwh)")
uploaded = st.file_uploader("Drag & drop or browse your CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # KPIs
    total_kwh = float(df["consumption_kwh"].sum())
    mean_kwh = float(df["consumption_kwh"].mean())
    max_kwh = float(df["consumption_kwh"].max())
    min_kwh = float(df["consumption_kwh"].min())
    total_cost = total_kwh * price_per_kwh

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total consumption", f"{total_kwh:,.2f} kWh")
    c2.metric("Average hourly", f"{mean_kwh:,.2f} kWh")
    c3.metric("Max hourly", f"{max_kwh:,.2f} kWh")
    c4.metric("Estimated cost", f"{total_cost:,.2f} €")

    st.subheader("Consumption chart")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["consumption_kwh"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    st.pyplot(fig)

    st.subheader("Daily summary")
    daily = df.set_index("timestamp").resample("D")["consumption_kwh"].sum().reset_index()
    st.dataframe(daily)

    # AI summary (placeholder until API key is added in Secrets)
    st.subheader("AI summary")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        ai_text = (
            "🎯 Demo summary (no API key detected):\n"
            f"• Total consumption {total_kwh:,.2f} kWh; average hourly {mean_kwh:,.2f} kWh.\n"
            f"• Estimated period cost: {total_cost:,.2f} €.\n"
            "• Consider shifting loads off-peak and optimizing HVAC/lighting during 09:00–18:00."
        )
    else:
        # Here is where you'd call the OpenAI API in production.
        ai_text = (
            "OPENAI_API_KEY found. In production, this will call the AI model and return a tailored summary. "
            "Showing demo text for now."
        )
    st.text_area("AI summary text", ai_text, height=160)

    # PDF export
    st.subheader("Export report")
    if st.button("Generate PDF"):
        stats = {"total_kwh": total_kwh, "mean_kwh": mean_kwh, "max_kwh": max_kwh, "min_kwh": min_kwh}
        tmp_path = "FluxTwin_Report.pdf"
        generate_pdf(tmp_path, project_name, stats, ai_text)
        with open(tmp_path, "rb") as f:
            st.download_button("Download PDF report", f, file_name="FluxTwin_Report.pdf", mime="application/pdf")
else:
    st.info("Upload the sample_data.csv or your own file to start the analysis.")
