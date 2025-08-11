from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime

def generate_pdf(path, project_name, stats, ai_summary):
    doc = SimpleDocTemplate(path, pagesize=A4, title=f"{project_name} Report")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{project_name} — Energy Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Generated: {generated}", styles['Normal']))
    story.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Total consumption (kWh)", f"{stats.get('total_kwh', 0):,.2f}"],
        ["Average hourly (kWh)", f"{stats.get('mean_kwh', 0):,.2f}"],
        ["Max hourly (kWh)", f"{stats.get('max_kwh', 0):,.2f}"],
        ["Min hourly (kWh)", f"{stats.get('min_kwh', 0):,.2f}"],
    ]
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EEEEEE")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 18))

    story.append(Paragraph("<b>AI Summary</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(ai_summary.replace("\n", "<br/>"), styles['BodyText']))

    doc.build(story)
    return path

    story.append(Paragraph("<b>Σύνοψη με AI</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(ai_summary.replace("\n", "<br/>"), styles['BodyText']))

    doc.build(story)
    return path
