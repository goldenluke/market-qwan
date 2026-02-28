from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


def generate_pdf(filename, metrics):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Market-QWAN Institutional Report", styles["Title"]))

    for k, v in metrics.items():
        elements.append(Paragraph(f"{k}: {round(v,4)}", styles["Normal"]))

    doc.build(elements)