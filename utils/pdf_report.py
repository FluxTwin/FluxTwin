# utils/pdf_report.py — Pro Report v1.2 (charts + daily forecast table)
from __future__ import annotations
__version__ = "pro-1.2"

import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image


# ---------- styles ----------
def _styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=ss["Title"], fontSize=20, leading=24, spaceAfter=10)
    h2    = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6)
    body  = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14)
    mono  = ParagraphStyle("Mono", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, textColor=colors.grey)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=12, bulletIndent=6, spaceAfter=4)
    return title, h2, body, mono, bullet


# ---------- plots ----------
def _plot_history_forecast_png(df: pd.DataFrame, forecast_df: pd.DataFrame | None) -> str:
    """Create a History vs Forecast chart as PNG and return its file path."""
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_png.close()
    path = tmp_png.name

    plt.figure(figsize=(8.6, 3.3))
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp")
    plt.plot(x["timestamp"], x["consumption_kwh"], label="History (consumption)", linewidth=1.6)
    if "production_kwh" in x.columns and x["production_kwh"].any():
        plt.plot(x["timestamp"], x["production_kwh"], label="History (production)", linewidth=1.2)

    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and "forecast_kwh" in forecast_df:
        f = forecast_df.copy()
        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"])
            plt.plot(f["date"], f["forecast_kwh"], label="Forecast (kWh)", linewidth=1.8)

    plt.xlabel("Date"); plt.ylabel("kWh"); plt.title("History vs Forecast"); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path


# ---------- main API ----------
def create_report(
    df: pd.DataFrame,
    advisor_text_or_list,
    price_eur_per_kwh: float = 0.25,
    expected_savings_pct: float = 0.12,
    forecast_df: pd.DataFrame | None = None,
    out_path: str = "FluxTwin_Report.pdf",
) -> str:
    """
    Enterprise-style PDF:
      - Header with version
      - KPIs (current period)
      - Cost analysis (forecast no-action vs after-actions)
      - Embedded chart (History vs Forecast)
      - Daily forecast table (Date | kWh | €) + totals
      - Actionable recommendations

    Συμβατό με παλιές κλήσεις: create_report(df, advisor_text).
    """

    title, h2, body, mono, bullet = _styles()

    # KPIs τρέχουσας περιόδου
    total = float(df["consumption_kwh"].sum())
    avg   = float(df["consumption_kwh"].mean()) if len(df) else 0.0
    mx    = float(df["consumption_kwh"].max()) if len(df) else 0.0
    mn    = float(df["consumption_kwh"].min()) if len(df) else 0.0
    baseline_cost = total * float(price_eur_per_kwh)

    # Forecast totals
    tot_fc = 0.0
    method = "—"
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df and not forecast_df.empty:
        tot_fc = float(forecast_df["forecast_kwh"].sum())
        method = str(forecast_df["method"].iloc[0]) if "method" in forecast_df.columns else "—"
    else:
        # fallback: naive mean από ημερήσια σύνολα
        x = df.copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"])
        day = x.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
        method = "naive-mean (fallback)"
        mean = float(day.mean()) if len(day) else 0.0
        tot_fc = mean * 7  # default horizon για quick cost σύγκριση
        forecast_df = pd.DataFrame({
            "date": pd.date_range(datetime.now().date(), periods=7, freq="D"),
            "forecast_kwh": [mean]*7,
            "method": ["naive-mean"]*7
        })

    cost_no_action = tot_fc * float(price_eur_per_kwh)
    cost_after     = cost_no_action * max(0.0, 1.0 - float(expected_savings_pct))
    savings_eur    = cost_no_action - cost_after

    # Tips -> σε λίστα
    if advisor_text_or_list is None:
        tips = []
    elif isinstance(advisor_text_or_list, (list, tuple)):
        tips = [str(t) for t in advisor_text_or_list if str(t).strip()]
    else:
        tips = [ln.strip("• ").strip() for ln in str(advisor_text_or_list).splitlines() if ln.strip()]

    # Build PDF
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    story = []

    # Header
    story += [
        Paragraph("FluxTwin — Energy Report", title),
        Paragraph("Version: Pro v1.2 (charts + daily forecast table)", mono),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", mono),
        Spacer(1, 6),
    ]

    # KPIs
    story += [Paragraph("Current period — KPIs", h2)]
    kpi_tbl = Table(
        [
            ["Metric", "Value"],
            ["Total consumption (kWh)", f"{total:,.2f}"],
            ["Average sample (kWh)", f"{avg:,.2f}"],
            ["Max sample (kWh)", f"{mx:,.2f}"],
            ["Min sample (kWh)", f"{mn:,.2f}"],
            ["Baseline cost (this period)", f"{baseline_cost:,.2f} €"],
        ],
        hAlign="LEFT",
        colWidths=[8.0*cm, 7.0*cm],
    )
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story += [kpi_tbl, Spacer(1, 10)]

    # Cost analysis
    story += [Paragraph("Forecast — cost impact", h2)]
    cost_tbl = Table(
        [
            ["Metric", "Value"],
            ["Forecast method", method],
            ["Forecast total (kWh)", f"{tot_fc:,.0f}"],
            ["Estimated cost (no action)", f"{cost_no_action:,.2f} €"],
            [f"Estimated cost (after actions, -{expected_savings_pct*100:.1f}%)", f"{cost_after:,.2f} €"],
            ["Estimated savings (€)", f"{savings_eur:,.2f} €"],
        ],
        hAlign="LEFT",
        colWidths=[9.0*cm, 6.0*cm],
    )
    cost_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story += [cost_tbl, Spacer(1, 10)]

    # Chart
    chart_path = _plot_history_forecast_png(df, forecast_df)
    story += [Paragraph("History vs Forecast", h2)]
    story += [Image(chart_path, width=16.5*cm, height=6.3*cm), Spacer(1, 10)]

    # Daily forecast table
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df and not forecast_df.empty:
        f = forecast_df.copy()
        if "date" in f.columns:
            f["Date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
        else:
            f["Date"] = [str(i+1) for i in range(len(f))]
        f["Forecast (kWh)"] = f["forecast_kwh"].astype(float)
        f["Cost (€)"] = f["Forecast (kWh)"] * float(price_eur_per_kwh)

        cols = ["Date", "Forecast (kWh)", "Cost (€)"]
        rows = [cols] + f[cols].round(2).astype(str).values.tolist()
        rows += [["Total", f"{f['Forecast (kWh)'].sum():,.2f}", f"{f['Cost (€)'].sum():,.2f} €"]]

        ftbl = Table(rows, hAlign="LEFT", colWidths=[5.0*cm, 5.0*cm, 5.0*cm])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.darkblue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,1), (-1,-1), "RIGHT"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story += [Paragraph("Daily forecast (kWh & €)", h2), ftbl, Spacer(1, 10)]

    # Recommendations
    story += [Paragraph("Next 7 Days — Action plan", h2)]
    if tips:
        for tip in tips:
            story.append(Paragraph(f"• {tip}", bullet))
    else:
        story.append(Paragraph("No recommendations available.", body))
    story += [Spacer(1, 8)]
    story += [Paragraph("Note: Savings are estimates based on profile and operational patterns.", mono)]

    doc.build(story)
    return out_path
