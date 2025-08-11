import os
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from utils.pdf_report import generate_pdf

load_dotenv()

st.set_page_config(page_title="FluxTwin – MVP", page_icon="⚡", layout="wide")

st.title("⚡ FluxTwin — Energy Analytics MVP")
st.caption("Πρώτη έκδοση: ανέβασε δεδομένα, δες ανάλυση, πάρε αναφορά PDF και σύνοψη με AI.")

with st.sidebar:
    st.header("Ρυθμίσεις")
    price_per_kwh = st.number_input("Τιμή ρεύματος (€ / kWh)", value=0.25, min_value=0.0, step=0.01)
    project_name = st.text_input("Όνομα έργου", value=os.getenv("FLUXTWIN_APP_NAME", "FluxTwin"))
    st.markdown("---")
    st.write("👈 Κατέβασε δείγμα δεδομένων για δοκιμή:")
    with open("assets/sample_data.csv", "rb") as f:
        st.download_button("Λήψη sample_data.csv", file_name="sample_data.csv", data=f, mime="text/csv")

uploaded = st.file_uploader("📂 Ανέβασε αρχείο CSV με στήλες: timestamp, consumption_kwh", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic stats
    total_kwh = float(df["consumption_kwh"].sum())
    mean_kwh = float(df["consumption_kwh"].mean())
    max_kwh = float(df["consumption_kwh"].max())
    min_kwh = float(df["consumption_kwh"].min())
    total_cost = total_kwh * price_per_kwh

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Συνολική κατανάλωση", f"{total_kwh:,.2f} kWh")
    col2.metric("Μέση ωριαία", f"{mean_kwh:,.2f} kWh")
    col3.metric("Μέγιστη ωριαία", f"{max_kwh:,.2f} kWh")
    col4.metric("Εκτιμώμενο κόστος", f"{total_cost:,.2f} €")

    st.subheader("Γράφημα κατανάλωσης")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["consumption_kwh"])
    ax.set_xlabel("Χρόνος")
    ax.set_ylabel("Κατανάλωση (kWh)")
    st.pyplot(fig)

    # Daily aggregation
    st.subheader("Ημερήσια σύνοψη")
    daily = df.set_index("timestamp").resample("D")["consumption_kwh"].sum().reset_index()
    st.dataframe(daily)

    # AI summary (placeholder if no API)
    st.subheader("Σύνοψη με AI")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        ai_text = (
            "🎯 Προσωρινή σύνοψη (χωρίς API key):\n"
            f"• Συνολική κατανάλωση {total_kwh:,.2f} kWh και μέση ωριαία {mean_kwh:,.2f} kWh.\n"
            f"• Εκτιμώμενο κόστος περιόδου: {total_cost:,.2f} €.\n"
            "• Εξετάστε μετατόπιση φορτίων εκτός αιχμής και βελτιστοποίηση HVAC/φωτισμού τις ώρες 09:00–18:00."
        )
    else:
        # We don't actually call external APIs here; provide instructions only.
        ai_text = (
            "Το κλειδί API εντοπίστηκε. Στην παραγωγή, εδώ θα γίνεται κλήση στο μοντέλο AI "
            "για προσαρμοσμένη σύνοψη και προτάσεις. "
            "Προς το παρόν εμφανίζεται demo κείμενο."
        )

    st.text_area("AI σύνοψη", ai_text, height=160)

    # Generate PDF
    st.subheader("Εξαγωγή αναφοράς")
    if st.button("Δημιουργία PDF"):
        pdf_bytes = io.BytesIO()
        tmp_path = "FluxTwin_Report.pdf"
        stats = {
            "total_kwh": total_kwh,
            "mean_kwh": mean_kwh,
            "max_kwh": max_kwh,
            "min_kwh": min_kwh,
        }
        generate_pdf(tmp_path, project_name, stats, ai_text)
        with open(tmp_path, "rb") as f:
            st.download_button("Λήψη PDF αναφοράς", f, file_name="FluxTwin_Report.pdf", mime="application/pdf")

else:
    st.info("Ανέβασε το sample_data.csv ή δικό σου αρχείο για να ξεκινήσεις την ανάλυση.")