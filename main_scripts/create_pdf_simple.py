#!/usr/bin/env python3
"""
Create PDF guide using fpdf2 - simplified version
"""

from fpdf import FPDF, XPos, YPos
import re

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('DejaVu', '', 8)
            self.cell(0, 10, 'Visual Toolkit 用户指南', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', align='C')


def create_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add Unicode fonts
    try:
        pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        pdf.add_font('DejaVu', 'B', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
        pdf.add_font('DejaVuMono', '', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    except Exception as e:
        print(f"Font error: {e}")
        return
    
    # Title page
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 20, 'Visual Toolkit', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 20, '神经元可视化工具包', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    pdf.set_font('DejaVu', '', 12)
    pdf.cell(0, 10, '用户指南 v1.0', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 10, '最后更新: 2026-03-20', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
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
                    pdf.cell(0, 5, code_line, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
            pdf.cell(0, 12, line[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.set_font('DejaVu', 'B', 14)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(0, 10, line[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        
        elif line.startswith('### '):
            pdf.ln(3)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.set_text_color(0, 76, 153)
            pdf.cell(0, 8, line[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
        
        elif line.startswith('#### '):
            pdf.ln(2)
            pdf.set_font('DejaVu', 'B', 11)
            pdf.cell(0, 6, line[5:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # ASCII diagrams - use smaller font
        elif any(c in line for c in ['┌', '│', '└', '├', '─']):
            pdf.set_font('DejaVuMono', '', 6)
            pdf.cell(0, 4, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Tables
        elif line.startswith('|') and '---' not in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells and cells[0] and not all('-' in c for c in cells):
                pdf.set_font('DejaVu', '', 9)
                cell_text = ' | '.join(cells[:3])
                pdf.cell(0, 5, cell_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Lists
        elif line.startswith('- ') or line.startswith('* '):
            pdf.set_font('DejaVu', '', 10)
            pdf.cell(10, 6, '')
            pdf.cell(5, 6, '•')
            # Handle remaining text
            text = line[2:]
            pdf.multi_cell(0, 6, text)
        
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
    print("✓ PDF generated: Visual_Toolkit_用户指南.pdf")


if __name__ == '__main__':
    create_pdf()
