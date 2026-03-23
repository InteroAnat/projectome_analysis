#!/usr/bin/env python3
"""
Create PDF guide using fpdf2 with Unicode support
"""

from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('DejaVu', '', 8)
            self.cell(0, 10, 'Visual Toolkit 用户指南', align='C')
            self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', align='C')


def create_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add Unicode fonts
    try:
        # Try to add DejaVu fonts (comes with fpdf2)
        pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', uni=True)
        pdf.add_font('DejaVuMono', '', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', uni=True)
    except:
        # Use built-in fonts as fallback
        pass
    
    # Title page
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 20, 'Visual Toolkit', align='C')
    pdf.ln(15)
    pdf.cell(0, 20, '神经元可视化工具包', align='C')
    pdf.ln(20)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, '用户指南 v1.0', align='C')
    pdf.ln(8)
    pdf.cell(0, 10, '最后更新: 2026-03-20', align='C')
    
    # Read markdown
    with open('VISUAL_TOOLKIT_GUIDE_CN.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    in_code = False
    code_buffer = []
    
    for line in lines:
        # Skip main title
        if line.startswith('# Visual Toolkit'):
            continue
        
        # Code blocks
        if line.startswith('```'):
            if in_code and code_buffer:
                pdf.set_font('DejaVuMono', '', 8)
                pdf.set_fill_color(240, 240, 240)
                for code_line in code_buffer:
                    pdf.cell(0, 5, code_line, ln=True, fill=True)
                pdf.ln(3)
                code_buffer = []
            in_code = not in_code
            continue
        
        if in_code:
            code_buffer.append(line)
            continue
        
        # Headers
        if line.startswith('# ') and not line.startswith('## '):
            pdf.add_page()
            pdf.set_font('DejaVu', 'B', 18)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(0, 12, line[2:], ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.set_font('DejaVu', 'B', 14)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(0, 10, line[3:], ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        
        elif line.startswith('### '):
            pdf.ln(3)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.set_text_color(0, 76, 153)
            pdf.cell(0, 8, line[4:], ln=True)
            pdf.set_text_color(0, 0, 0)
        
        elif line.startswith('#### '):
            pdf.ln(2)
            pdf.set_font('DejaVu', 'B', 11)
            pdf.cell(0, 6, line[5:], ln=True)
        
        # ASCII diagrams
        elif '┌' in line or '│' in line or '└' in line or '├' in line:
            pdf.set_font('DejaVuMono', '', 7)
            pdf.cell(0, 4, line, ln=True)
        
        # Tables (simplified)
        elif line.startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells and cells[0] and not all('-' in c for c in cells):
                pdf.set_font('DejaVu', '', 9)
                cell_text = ' | '.join(cells[:3])
                pdf.cell(0, 5, cell_text, ln=True)
        
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            pdf.set_font('DejaVu', '', 10)
            pdf.cell(5, 6, '', ln=0)
            pdf.cell(5, 6, '•', ln=0)
            pdf.multi_cell(0, 6, line[2:])
        
        elif re.match(r'^\d+\. ', line):
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 6, line)
        
        # Empty lines
        elif not line.strip():
            pdf.ln(3)
        
        # Regular text
        else:
            pdf.set_font('DejaVu', '', 10)
            pdf.multi_cell(0, 6, line)
    
    # Save
    pdf.output('Visual_Toolkit_用户指南.pdf')
    print("PDF generated successfully: Visual_Toolkit_用户指南.pdf")


if __name__ == '__main__':
    create_pdf()
