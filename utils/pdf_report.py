# utils/pdf_report.py
from __future__ import annotations
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def _styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=ss["Title"], fontSize=20, leading=24, spaceAfter=10)
    h2    = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6)
    body  = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14)
    mono  = ParagraphStyle("Mono", parent=ss["BodyText"], fontName="Helvetica", fontSize=9)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=12, bulletIndent=6, spaceAfter=4)
    return title, h2, body, mono, bullet

def create_report(
    df: pd.DataFrame,
    advisor_text_or_list,
    price_eur_per_kwh: float = 0.25,
    expected_savings_pct: float = 0.12,
    forecast_df: pd.DataFrame | None = None,
    out_path: str = "FluxTwin_Report.pdf",
) -> str:
    """Δημιουργεί enterprise-style PDF με KPIs, Cost analysis, Forecast summary, Recommendations."""
    title, h2, body, mono, bullet = _styles()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    story = []

    # Header
    story += [
        Paragraph("FluxTwin — Energy Report", title),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", mono),
        Spacer(1, 6),
    ]

    # KPIs
    total = float(df["consumption_kwh"].sum())
    avg   = float(df["consumption_kwh"].mean()) if len(df) else 0.0
    mx    = float(df["consumption_kwh"].max()) if len(df) else 0.0
    mn    = float(df["consumption_kwh"].min()) if len(df) else 0.0

    baseline_cost  = total * float(price_eur_per_kwh)
    projected_cost = baseline_cost * max(0.0, 1.0 - float(expected_savings_pct))
    savings_eur    = baseline_cost - projected_cost

    story += [Paragraph("Key metrics", h2)]
    kpi_tbl = Table(
        [
            ["Metric", "Value"],
            ["Total consumption (kWh)", f"{total:,.2f}"],
            ["Average sample (kWh)", f"{avg:,.2f}"],
            ["Max sample (kWh)", f"{mx:,.2f}"],
            ["Min sample (kWh)", f"{mn:,.2f}"],
        ],
        hAlign="LEFT",
        colWidths=[7*cm, 8*cm],
    )
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    story += [kpi_tbl, Spacer(1, 10)]

    # Costs
    story += [Paragraph("Cost analysis", h2)]
    cost_tbl = Table(
        [
            ["Electricity price (€/kWh)", f"{price_eur_per_kwh:.3f} €"],
            ["Baseline cost (this period)", f"{baseline_cost:,.2f} €"],
            [f"Projected cost (−{expected_savings_pct*100:.1f}% savings)", f"{projected_cost:,.2f} €"],
            ["Estimated savings", f"{savings_eur:,.2f} €"],
        ],
        hAlign="LEFT",
        colWidths=[9*cm, 6*cm],
    )
    cost_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
    ]))
    story += [cost_tbl, Spacer(1, 10)]

    # Forecast (optional)
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df:
        story += [Paragraph("7–30 day forecast (summary)", h2)]
        tot_fc = float(forecast_df["forecast_kwh"].sum())
        fc_cost = tot_fc * float(price_eur_per_kwh)
        story += [Paragraph(
            f"Projected energy for horizon: <b>{tot_fc:,.0f} kWh</b> "
            f" (~{fc_cost:,.2f} € at {price_eur_per_kwh:.3f} €/kWh).",
            body
        ), Spacer(1, 6)]

    # Recommendations
    story += [Paragraph("Actionable recommendations", h2)]
    if isinstance(advisor_text_or_list, (list, tuple)):
        for tip in advisor_text_or_list:
            story.append(Paragraph(f"• {tip}", bullet))
    else:
        for line in str(advisor_text_or_list).splitlines():
            if line.strip():
                story.append(Paragraph(f"• {line.strip()}", bullet))
    story += [Spacer(1, 10)]

    story += [Paragraph(
        "Note: Savings are estimates based on profile and operational patterns. "
        "For higher accuracy, enable live data streaming and tariff-aware optimization.",
        mono
    )]

    doc.build(story)
    return out_path
