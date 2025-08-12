# utils/pdf_report.py
from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def _build_kpis_table(df: pd.DataFrame) -> Table:
    total_kwh = float(df["consumption_kwh"].sum())
    avg_kwh   = float(df["consumption_kwh"].mean())
    max_kwh   = float(df["consumption_kwh"].max())
    min_kwh   = float(df["consumption_kwh"].min())
    rows = [
        ["Metric", "Value"],
        ["Total consumption (kWh)", f"{total_kwh:,.2f}"],
        ["Average sample (kWh)",   f"{avg_kwh:,.2f}"],
        ["Max sample (kWh)",       f"{max_kwh:,.2f}"],
        ["Min sample (kWh)",       f"{min_kwh:,.2f}"],
    ]
    tbl = Table(rows, colWidths=[8*cm, 6*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F4F7")),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor("#D0D5DD")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
    ]))
    return tbl

def _build_chart_image(df: pd.DataFrame) -> BytesIO:
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=140)
    ax.plot(df["timestamp"], df["consumption_kwh"], label="Consumption (kWh)")
    if "production_kwh" in df.columns:
        ax.plot(df["timestamp"], df["production_kwh"], label="Production (kWh)")
    ax.set_xlabel("Time"); ax.set_ylabel("kWh (per sample)")
    ax.grid(True, alpha=0.25); ax.legend(loc="upper right")
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    return buf

def create_report(df: pd.DataFrame, advice_text: str, out_path: str = "FluxTwin_Report.pdf") -> str:
    df = df.copy(); df["timestamp"] = pd.to_datetime(df["timestamp"])
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    story = []
    story.append(Paragraph("<b>FluxTwin — Energy Report</b>", styles["Title"]))
    story.append(Spacer(1, 6)); story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Small"]))
    story.append(Spacer(1, 12))
    story.append(_build_kpis_table(df)); story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Advisor recommendation</b>", styles["Heading2"]))
    story.append(Paragraph(advice_text.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Consumption chart</b>", styles["Heading2"]))
    chart_buf = _build_chart_image(df)
    story.append(Image(chart_buf, width=16.5*cm, height=8.0*cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Daily summary</b>", styles["Heading2"]))
    daily = df.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    rows = [["Date", "Consumption (kWh)"]] + [[d.strftime("%Y-%m-%d"), f"{kwh:,.2f}"] for d, kwh in daily.items()]
    day_tbl = Table(rows, colWidths=[7*cm, 6*cm])
    day_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F4F7")),
        ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#E4E7EC")),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
    ]))
    story.append(day_tbl)
    doc.build(story)
    return out_path
