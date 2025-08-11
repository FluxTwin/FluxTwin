from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime
from io import BytesIO

def generate_pdf(path, project_name, stats, ai_summary,
                 chart_png_bytes=None, daily_rows=None,
                 logo_png_path=None,
                 forecast_png_bytes=None, forecast_rows=None):
    doc = SimpleDocTemplate(
        path, pagesize=A4, title=f"{project_name} Report",
        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))

    story = []

    # Header (optional logo)
    if logo_png_path:
        try:
            story.append(Image(logo_png_path, width=2.2*cm, height=2.2*cm))
            story.append(Spacer(1, 6))
        except Exception:
            pass

    story.append(Paragraph(f"<b>{project_name} — Energy Report</b>", styles['Title']))
    story.append(Spacer(1, 6))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Generated: {generated}", styles['Small']))
    story.append(Spacer(1, 12))

    # ---- KPI table (now includes forecast KPIs if present) ----
    data = [
        ["Metric", "Value"],
        ["Total consumption (kWh)", f"{stats.get('total_kwh', 0):,.2f}"],
        ["Average hourly (kWh)", f"{stats.get('mean_kwh', 0):,.2f}"],
        ["Max hourly (kWh)", f"{stats.get('max_kwh', 0):,.2f}"],
        ["Min hourly (kWh)", f"{stats.get('min_kwh', 0):,.2f}"],
        ["Estimated cost (€)", f"{stats.get('total_cost_eur', 0):,.2f}"],
    ]

    # Add forecast KPIs if provided
    if "forecast_total_kwh" in stats:
        data.append(["Forecast total (kWh)", f"{stats['forecast_total_kwh']:,.2f}"])
    if "forecast_cost_eur" in stats:
        data.append(["Estimated cost (forecast) (€)", f"{stats['forecast_cost_eur']:,.2f}"])

    tbl = Table(data, hAlign="LEFT", colWidths=[8*cm, 5*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F4F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D0D5DD")),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # ---- AI summary ----
    story.append(Paragraph("<b>AI Summary</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    clean = (ai_summary or "").replace("\n•", "<br/>•").replace("\n- ", "<br/>• ")
    story.append(Paragraph(clean.replace("\n", "<br/>"), styles['BodyText']))
    story.append(Spacer(1, 12))

    # ---- Consumption chart ----
    if chart_png_bytes:
        story.append(Paragraph("<b>Consumption chart</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(Image(BytesIO(chart_png_bytes), width=16.5*cm, height=8.5*cm))
        story.append(Spacer(1, 12))

    # ---- Daily summary table ----
    if daily_rows:
        story.append(Paragraph("<b>Daily summary</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        rows = [["Date", "Consumption (kWh)"]] + daily_rows
        day_tbl = Table(rows, hAlign="LEFT", colWidths=[7*cm, 6*cm])
        day_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F4F7")),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#E4E7EC")),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ]))
        story.append(day_tbl)
        story.append(Spacer(1, 12))

    # ---- Forecast section ----
    if forecast_png_bytes or forecast_rows:
        story.append(Paragraph("<b>7–30 day Forecast</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

    if forecast_png_bytes:
        story.append(Image(BytesIO(forecast_png_bytes), width=16.5*cm, height=8.5*cm))
        story.append(Spacer(1, 12))

    if forecast_rows:
        rows = [["Date", "Forecast (kWh)"]] + forecast_rows
        f_tbl = Table(rows, hAlign="LEFT", colWidths=[7*cm, 6*cm])
        f_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F2F4F7")),
            ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#E4E7EC")),
            ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ]))
        story.append(f_tbl)

    doc.build(story)
    return path
