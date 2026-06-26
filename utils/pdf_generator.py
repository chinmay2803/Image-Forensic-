from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                 Image as RLImage, Table, TableStyle, PageBreak)
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime
import os

def generate_forensic_report(image_path, result, techniques, metadata, qtables, visuals_paths):
    """
    Generate a comprehensive PDF forensic report.
    Returns BytesIO buffer containing the PDF.
    """
    # Create BytesIO buffer
    buffer = BytesIO()
    
    # Create PDF document with buffer
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        topMargin=0.5*inch, 
        bottomMargin=0.5*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("🔬 AutoSplice Forensic Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Report Info Box
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_data = [
        ['Report Generated:', report_date],
        ['Image File:', os.path.basename(image_path)],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Detection Result
    story.append(Paragraph("📊 Detection Result", heading_style))
    result_color = colors.green if 'REAL' in result or 'No strong evidence' in result else colors.red
    result_text = f"<font color='{result_color.hexval()}' size='12'><b>{result}</b></font>"
    story.append(Paragraph(result_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Triggered Techniques
    if techniques:
        story.append(Paragraph("🎯 Detection Techniques Triggered", heading_style))
        for tech in techniques:
            story.append(Paragraph(f"• {tech}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Metadata Table
    if metadata and len(metadata) > 0:
        story.append(Paragraph("📷 Image Forensic Information", heading_style))
        metadata_data = [['Property', 'Value']]
        
        # Limit to first 12 items to avoid overflow
        count = 0
        for key, value in metadata.items():
            if count >= 12:
                break
            metadata_data.append([str(key)[:30], str(value)[:70]])
            count += 1
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Q-Tables
    if qtables and 'info' not in qtables and 'error' not in qtables:
        story.append(Paragraph("🔢 Quantization Analysis", heading_style))
        for key, value in list(qtables.items())[:5]:  # Limit items
            story.append(Paragraph(f"<b>{key}:</b> {str(value)[:100]}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Visual Analysis - New page for images
    if visuals_paths:
        story.append(PageBreak())
        story.append(Paragraph("🔍 Forensic Visual Analysis", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        for key, img_path in visuals_paths.items():
            if os.path.exists(img_path):
                try:
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}</b>", styles['Normal']))
                    img = RLImage(img_path, width=3.5*inch, height=2.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    print(f"Could not add image {key}: {e}")
                    story.append(Paragraph(f"<i>Image {key} could not be loaded</i>", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("_" * 100, styles['Normal']))
    footer_style = ParagraphStyle(
        'footer',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph(
        "<i>Generated by AutoSplice - AI-Powered Forensic Detection System</i>", 
        footer_style
    ))
    
    # Build PDF - THIS WRITES TO BUFFER
    doc.build(story)
    
    # CRITICAL: Reset buffer pointer to beginning
    buffer.seek(0)
    
    return buffer
