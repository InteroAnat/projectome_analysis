#!/usr/bin/env python3
"""
Create PDF guide for Visual Toolkit (Chinese)
Using reportlab with CJK font support
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab.rl_config
reportlab.rl_config.warnOnMissingFontGlyphs = 0
import re

# Register Chinese fonts
try:
    pdfmetrics.registerFont(TTFont('NotoSansSC', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'))
    pdfmetrics.registerFont(TTFont('NotoSansSC-Bold', '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc'))
    # Map to family
    from reportlab.lib.fonts import addMapping
    addMapping('NotoSansSC', 0, 0, 'NotoSansSC')  # normal
    addMapping('NotoSansSC', 1, 0, 'NotoSansSC-Bold')  # bold
except Exception as e:
    print(f"Font registration warning: {e}")

# Fallback to standard fonts
DEFAULT_FONT = 'NotoSansSC' if 'NotoSansSC' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
DEFAULT_FONT_BOLD = 'NotoSansSC-Bold' if 'NotoSansSC-Bold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'

def create_pdf():
    doc = SimpleDocTemplate(
        "Visual_Toolkit_用户指南.pdf",
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#003366'),
        spaceAfter=30,
        alignment=1,  # Center
        fontName=DEFAULT_FONT_BOLD
    )
    
    heading1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12,
        fontName=DEFAULT_FONT_BOLD
    )
    
    heading2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#004d99'),
        spaceAfter=10,
        fontName=DEFAULT_FONT_BOLD
    )
    
    heading3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=8,
        fontName=DEFAULT_FONT_BOLD
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        fontName=DEFAULT_FONT
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leading=10,
        fontName='Courier',
        backColor=colors.HexColor('#f5f5f5'),
        leftIndent=10
    )
    
    # Build content
    story = []
    
    # Title page
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph("Visual Toolkit", title_style))
    story.append(Paragraph("神经元可视化工具包", title_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("用户指南 v1.0", heading2_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("最后更新: 2026-03-20", body_style))
    story.append(PageBreak())
    
    # Read and parse markdown
    with open('VISUAL_TOOLKIT_GUIDE_CN.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    in_code = False
    code_buffer = []
    
    for line in lines:
        # Skip title (already on cover)
        if line.startswith('# Visual Toolkit'):
            continue
            
        # Code blocks
        if line.startswith('```'):
            if in_code and code_buffer:
                code_text = '<br/>'.join(code_buffer)
                story.append(Preformatted(code_text, code_style))
                story.append(Spacer(1, 0.3*cm))
                code_buffer = []
            in_code = not in_code
            continue
        
        if in_code:
            # Escape HTML
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            code_buffer.append(escaped)
            continue
        
        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            story.append(PageBreak())
            story.append(Paragraph(line[2:], heading1_style))
            story.append(Spacer(1, 0.2*cm))
        
        elif line.startswith('## '):
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(line[3:], heading2_style))
            story.append(Spacer(1, 0.1*cm))
        
        elif line.startswith('### '):
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(line[4:], heading3_style))
        
        elif line.startswith('#### '):
            story.append(Spacer(1, 0.1*cm))
            story.append(Paragraph(line[5:], heading3_style))
        
        # Tables
        elif line.startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells and cells[0]:
                # Simple table representation
                table_text = ' | '.join(cells[:4])
                story.append(Paragraph(table_text, body_style))
        
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph('• ' + line[2:], body_style))
        
        elif re.match(r'^\d+\. ', line):
            story.append(Paragraph(line, body_style))
        
        # Empty lines
        elif not line.strip():
            story.append(Spacer(1, 0.2*cm))
        
        # Regular text
        else:
            # Handle inline code
            text = line.replace('`', '<code>')
            # Simple bold
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(text, body_style))
    
    # Build PDF
    doc.build(story)
    print("PDF created: Visual_Toolkit_用户指南.pdf")


if __name__ == '__main__':
    create_pdf()
