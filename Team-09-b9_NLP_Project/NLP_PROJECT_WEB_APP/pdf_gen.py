from fpdf import FPDF
from io import BytesIO
from datetime import datetime

def generate_pdf_report(transcript, summary, action_items, sentiment, keywords):
    """Generate PDF report of meeting minutes"""
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "MEETING MINUTES REPORT", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    
    # Summary Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "SUMMARY", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 4, summary if summary else "No summary generated")
    pdf.ln(3)
    
    # Sentiment
    if sentiment:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "SENTIMENT ANALYSIS", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, sentiment.encode('latin-1', 'ignore').decode('latin-1'), ln=True)
        pdf.ln(3)
    
    # Keywords
    if keywords:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "KEY TOPICS", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 4, ", ".join(keywords))
        pdf.ln(3)
    
    # Action Items Table
    if action_items:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "ACTION ITEMS", ln=True)
        pdf.set_font("Arial", "", 9)
        
        # Table header
        pdf.set_fill_color(42, 82, 152)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 6, "Person", border=1, fill=True)
        pdf.cell(80, 6, "Task", border=1, fill=True)
        pdf.cell(30, 6, "Deadline", border=1, fill=True)
        pdf.cell(30, 6, "Priority", border=1, fill=True)
        pdf.ln()
        
        # Table data
        pdf.set_text_color(0, 0, 0)
        for item in action_items:
            pdf.cell(40, 6, item['Person'][:15], border=1)
            pdf.cell(80, 6, item['Task'][:40], border=1)
            pdf.cell(30, 6, item['Deadline'], border=1)
            pdf.cell(30, 6, item['Priority'][:10].encode('latin-1', 'ignore').decode('latin-1'), border=1)
            pdf.ln()
        
        pdf.ln(3)
    
    # Transcript Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "FULL TRANSCRIPT", ln=True)
    pdf.set_font("Arial", "", 9)
    
    if len(transcript) > 2000:
        transcript_display = transcript[:2000] + "..."
    else:
        transcript_display = transcript
    
    pdf.multi_cell(0, 3, transcript_display)
    
    # Generate PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes
