# utils/pdf_report.py
from __future__ import annotations
import os
import io
import tempfile
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)

# ---------------------- helpers ----------------------
def _styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle("TitleX", parent=ss["Title"], fontSize=20, leading=24, spaceAfter=12)
    h2    = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=6)
    body  = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14)
    mono  = ParagraphStyle("Mono", parent=ss["BodyText"], fontName="Helvetica", fontSize=9)
    return title, h2, body, mono

def _ensure_daily(df: pd.DataFrame) -> pd.Series:
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    day = x.set_index("timestamp")["consumption_kwh"].resample("D").sum().dropna()
    return day

def _plot_history_chart(df: pd.DataFrame) -> str:
    # δημιουργεί PNG στο temp και επιστρέφει το path
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df["timestamp"], df["consumption_kwh"], label="Consumption (kWh)")
    if "production_kwh" in df.columns and df["production_kwh"].any():
        ax.plot(df["timestamp"], df["production_kwh"], label="Production (kWh)")
    ax.set_xlabel("Time"); ax.set_ylabel("kWh"); ax.set_title("Consumption vs Production")
    ax.legend(); fig.tight_layout()
    tmp = os.path.join(tempfile.gettempdir(), f"ft_hist_{datetime.now().timestamp()}.png")
    fig.savefig(tmp, dpi=150); plt.close(fig)
    return tmp

def _mk_table(data, col_widths=None, header_bg=colors.lightgrey):
    t = Table(data, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
    ]))
    return t

def _coerce_tips(advisor_text_or_list) -> list[str]:
    if advisor_text_or_list is None:
        return []
    if isinstance(advisor_text_or_list, (list, tuple)):
        return [str(x) for x in advisor_text_or_list if str(x).strip()]
    txt = str(advisor_text_or_list)
    return [ln.strip("• ").strip() for ln in txt.splitlines() if ln.strip()]

# ---------------------- public API ----------------------
def create_report(
    df: pd.DataFrame,
    advisor_text_or_list,
    price_eur_per_kwh: float = 0.25,
    expected_savings_pct: float | None = None,
    forecast_df: pd.DataFrame | None = None,
) -> str:
    """
    Συμβατότητα προς τα πίσω:
      - παλιό: create_report(df, advisor_text)
      - νέο:  create_report(df, tips/list, price, expected_savings_pct, forecast_df)

    Επιστρέφει path του PDF.
    """
    title, h2, body, mono = _styles()

    # --- prepare numbers ---
    total = float(df["consumption_kwh"].sum())
    avg   = float(df["consumption_kwh"].mean()) if len(df) else 0.0
    mx    = float(df["consumption_kwh"].max()) if len(df) else 0.0
    mn    = float(df["consumption_kwh"].min()) if len(df) else 0.0
    baseline_cost = total * float(price_eur_per_kwh)

    # forecast aggregation (αν δεν μας έδωσαν forecast_df, βγάζουμε απλό baseline-based)
    if isinstance(forecast_df, pd.DataFrame) and "forecast_kwh" in forecast_df:
        fc = forecast_df.copy()
        method = str(fc["method"].iloc[0]) if "method" in fc.columns and len(fc) else "n/a"
        tot_fc = float(fc["forecast_kwh"].sum())
    else:
        day = _ensure_daily(df)
        method = "naive-mean (fallback)"
        mean = float(day.mean()) if len(day) else 0.0
        # default horizon: 7 & 30
        tot_fc_7 = mean * 7
        tot_fc_30 = mean * 30
        # θα φτιάξουμε πίνακα 7 & 30 από το mean (αν δεν υπάρχει forecast_df)
        fc = pd.DataFrame({
            "date": [f"7 days (mean)", f"30 days (mean)"],
            "forecast_kwh": [tot_fc_7, tot_fc_30],
            "method": ["naive-mean", "naive-mean"]
        })
        tot_fc = tot_fc_7  # για το “no action” quick calc

    est_cost_no_action = tot_fc * float(price_eur_per_kwh)
    # savings %
    if expected_savings_pct is None:
        expected_savings_pct = 0.12
    est_cost_after = est_cost_no_action * (1.0 - float(expected_savings_pct))
    savings_eur = est_cost_no_action - est_cost_after

    tips = _coerce_tips(advisor_text_or_list)

    # --- build PDF ---
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp.name
    tmp.close()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []

    # Header
    story += [
        Paragraph("FluxTwin — Executive Energy Report", title),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", mono),
        Spacer(1, 8),
    ]

    # KPIs table
    story += [Paragraph("Current period — KPIs", h2)]
    kpi_tbl = _mk_table([
        ["Metric", "Value"],
        ["Total consumption (kWh)", f"{total:,.2f}"],
        ["Average sample (kWh)", f"{avg:,.2f}"],
        ["Max sample (kWh)", f"{mx:,.2f}"],
        ["Min sample (kWh)", f"{mn:,.2f}"],
        ["Estimated cost (this period)", f"{baseline_cost:,.2f} €"],
    ], col_widths=[9*cm, 6*cm])
    story += [kpi_tbl, Spacer(1, 10)]

    # History chart
    try:
        img_path = _plot_history_chart(df)
        story += [Paragraph("Consumption & Production overview", h2),
                  Image(img_path, width=16*cm, height=7*cm),
                  Spacer(1, 10)]
    except Exception as _:
        story += [Paragraph("Chart unavailable (matplotlib/pillow missing).", body), Spacer(1, 10)]

    # Forecast section
    story += [Paragraph("Forecast window — cost outlook", h2)]
    if "date" in fc.columns:
        # Make a clean table of forecast rows
        # Αν έχουν έρθει ημερομηνίες pandas, τις κάνουμε string
        fc_show = fc.copy()
        if pd.api.types.is_datetime64_any_dtype(fc_show["date"]):
            fc_show["date"] = fc_show["date"].dt.strftime("%Y-%m-%d")
        # προσθέτουμε κόστος ανά σειρά (όπου γίνεται)
        def _row_cost(v):
            try:
                return float(v) * float(price_eur_per_kwh)
            except Exception:
                return ""
        cost_col = fc_show["forecast_kwh"].apply(_row_cost)
        rows = [["Date/Period", "Forecast (kWh)", "Estimated cost (€)"]]
        for i in range(len(fc_show)):
            rows.append([str(fc_show["date"].iloc[i]), f"{float(fc_show['forecast_kwh'].iloc[i]):,.2f}", 
                         f"{cost_col.iloc[i]:,.2f} €" if cost_col.iloc[i] != "" else ""])
        fc_tbl = _mk_table(rows, col_widths=[7*cm, 5*cm, 3*cm], header_bg=colors.whitesmoke)
        story += [Paragraph(f"Method: <b>{method}</b>", body), fc_tbl, Spacer(1, 10)]
    else:
        story += [Paragraph("No forecast data available.", body), Spacer(1, 10)]

    # Cost summary (no action vs after actions)
    cost_tbl = _mk_table([
        ["Metric", "Value"],
        ["Estimated cost (no action)", f"{est_cost_no_action:,.2f} €"],
        [f"Estimated cost (after actions, -{float(expected_savings_pct)*100:.1f}%)", f"{est_cost_after:,.2f} €"],
        ["Estimated savings", f"{savings_eur:,.2f} €"],
    ], col_widths=[9*cm, 6*cm], header_bg=colors.darkblue)
    cost_tbl.setStyle(TableStyle([("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke)]))
    story += [Paragraph("Cost impact", h2), cost_tbl, Spacer(1, 10)]

    # Recommendations
    story += [Paragraph("Next 7 days — Action plan", h2)]
    if tips:
        for t in tips:
            story.append(Paragraph(f"• {t}", body))
    else:
        story.append(Paragraph("No recommendations available.", body))
    story += [Spacer(1, 8),
              Paragraph("Note: Savings are indicative, based on profile & operational patterns.", mono)]

    doc.build(story)
    return pdf_path
