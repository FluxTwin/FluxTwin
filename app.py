import streamlit as st
import pandas as pd
import os
import time
from utils import advisor, pdf_report
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="FluxTwin - Live Energy Advisor", layout="wide")

# ---------- DATA LOADING ----------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# ---------- MODES ----------
st.sidebar.title("Data Mode")
mode = st.sidebar.selectbox("Select mode", ["Upload CSV", "Live simulation (in-app)", "Watch realtime CSV (local)"])

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
    else:
        data = pd.DataFrame()

elif mode == "Live simulation (in-app)":
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = pd.DataFrame(columns=["timestamp", "consumption_kwh", "production_kwh"])
    if st.button("Add tick (new reading)"):
        now = datetime.now()
        cons = round(2 + (4 * os.urandom(1)[0] / 255), 2)
        prod = round(4 * os.urandom(1)[0] / 255, 2)
        new_row = pd.DataFrame([[now, cons, prod]], columns=["timestamp", "consumption_kwh", "production_kwh"])
        st.session_state.sim_data = pd.concat([st.session_state.sim_data, new_row], ignore_index=True)
    data = st.session_state.sim_data

elif mode == "Watch realtime CSV (local)":
    path = st.text_input("Path to realtime CSV", value="realtime_data.csv")
    if st.button("Refresh now"):
        data = load_data(path)
    else:
        data = pd.DataFrame()

# ---------- DISPLAY ----------
st.title("FluxTwin - Live Energy Advisor")

if not data.empty:
    st.subheader("📊 Live Data Overview")
    latest = data.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Consumption (kWh)", latest['consumption_kwh'])
    col2.metric("Current Production (kWh)", latest['production_kwh'])
    col3.metric("Net Usage (kWh)", round(latest['consumption_kwh'] - latest['production_kwh'], 2))

    # Advisor
    suggestion = advisor.get_advice(latest['consumption_kwh'], latest['production_kwh'])
    st.info(f"💡 Advisor: {suggestion}")

    # Plot
    fig = px.line(data, x="timestamp", y=["consumption_kwh", "production_kwh"], title="Consumption vs Production")
    st.plotly_chart(fig, use_container_width=True)

    # Export PDF
    if st.button("Generate PDF Report"):
        pdf_file = pdf_report.create_report(data, suggestion)
        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name="FluxTwin_Report.pdf")
else:
    st.warning("No data to display yet.")


