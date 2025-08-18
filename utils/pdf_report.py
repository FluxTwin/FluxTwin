# utils/pdf_report.py
import os
from datetime import datetime
import tempfile

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

# ---------- HELPER: plot chart to image ----------
def save_chart_as_img(df: pd.DataFrame, path: str):
    plt.figure(figsize=(7, 3))
    plt.plot(df["timestamp"], df["consumption_kwh"], label="Consumption (kWh)", color="red")
    if "production_kwh" in df.columns and df["production_kwh"].any():
        plt.plot(df["timestamp"], df["production_kwh"], label="Production (kWh)", color="green")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    plt.title("Consumption vs Production")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------- CREATE REPORT ----------
def create_report(df: pd.DataFrame, advisor_text: str, price_per_kwh: float = 0.25) -> str:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp_file.name
    tmp_file.close()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal = styles["Normal"]
    subtitle = ParagraphStyle("Subtitle", parent=styles["Heading2"], spaceAfter=10)

    # ---------- Title ----------
    story.append(Paragraph("⚡ FluxTwin — Energy Efficiency Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 24))

    # ---------- KPIs ----------
    total = float(df["consumption_kwh"].sum())
    avg = float(df["consumption_kwh"].mean())
    mx = float(df["consumption_kwh"].max())
    est_cost = total * price_per_kwh

    kpi_data = [
        ["Metric", "Value"],
        ["Total Consumption", f"{total:,.2f} kWh"],
        ["Average per sample", f"{avg:,.2f} kWh"],
        ["Max consumption", f"{mx:,.2f} kWh"],
        ["Estimated Cost", f"{est_cost:,.2f} €"],
    ]
    table = Table(kpi_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(Paragraph("Key Performance Indicators", subtitle))
    story.append(table)
    story.append(Spacer(1, 24))

    # ---------- Chart ----------
    story.append(Paragraph("Consumption & Production Overview", subtitle))
    chart_path = os.path.join(tempfile.gettempdir(), "chart.png")
    save_chart_as_img(df, chart_path)
    story.append(Image(chart_path, width=16 * cm, height=7 * cm))
    story.append(Spacer(1, 24))

    # ---------- Forecast ----------
    story.append(Paragraph("Forecast (7–30 days)", subtitle))
    daily = df.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    if len(daily) > 2:
        baseline = daily.mean()
        forecast_7 = baseline * 7
        forecast_30 = baseline * 30

        forecast_cost_7 = forecast_7 * price_per_kwh
        forecast_cost_30 = forecast_30 * price_per_kwh

        forecast_data = [
            ["Period", "Forecast kWh", "Forecast Cost (€)"],
            ["7 days", f"{forecast_7:,.2f}", f"{forecast_cost_7:,.2f} €"],
            ["30 days", f"{forecast_30:,.2f}", f"{forecast_cost_30:,.2f} €"],
        ]
        ftable = Table(forecast_data, hAlign="LEFT")
        ftable.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(ftable)
    else:
        story.append(Paragraph("⚠ Not enough data for forecast", normal))
    story.append(Spacer(1, 24))

    # ---------- Advisor ----------
    story.append(Paragraph("Energy Saving Recommendations", subtitle))
    story.append(Paragraph(advisor_text, normal))
    story.append(Spacer(1, 12))

    # ---------- Wrap up ----------
    story.append(Paragraph(
        "This report was generated automatically by FluxTwin Energy Advisor. "
        "For best results, consider combining these recommendations with smart energy management systems.",
        normal
    ))

    doc.build(story)
    return pdf_path
