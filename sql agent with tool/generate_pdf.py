import os
import urllib.request
from fpdf import FPDF
from PIL import Image, ImageDraw

# Import run_agent from sql_agent
from sql_agent import run_agent

# Paths
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGO_PATH = os.path.join(WORKSPACE_DIR, "mru_logo.png")
PDF_PATH = os.path.join(WORKSPACE_DIR, "sql_agent_experiment.pdf")

def prepare_logo():
    """Downloads the university logo or generates a clean high-res vector-style fallback image."""
    logo_url = "https://static.wixstatic.com/media/6685d7_3d4e7a3b47f645e6b8cada0ebadaae63~mv2.png"
    try:
        print(f"Downloading Malla Reddy University logo from: {logo_url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(logo_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(LOGO_PATH, 'wb') as f:
                f.write(response.read())
        print("Logo downloaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to download logo ({e}). Generating high-quality fallback logo image.")
        # Create a beautiful fallback image matching Malla Reddy University branding
        img = Image.new('RGB', (400, 150), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        # Burgundy borders and crest placeholder
        d.rectangle([(10, 10), (390, 140)], outline=(115, 25, 44), width=4)
        d.rectangle([(20, 20), (70, 70)], outline=(115, 25, 44), width=3)
        d.text((32, 38), "MRU", fill=(115, 25, 44))
        
        # Text branding
        d.text((90, 35), "MALLA REDDY UNIVERSITY", fill=(115, 25, 44))
        d.text((90, 55), "Maisammaguda, Hyderabad, India", fill=(100, 100, 100))
        d.text((90, 75), "ESTD - 2020", fill=(128, 128, 128))
        img.save(LOGO_PATH)
        return True

class ReportPDF(FPDF):
    def header(self):
        # Omit header on cover page (page 1)
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'Experiment 4: SQL Agent with Tool Use (ReAct)', border=0, align='R')
            self.ln(12)
            
    def footer(self):
        # Omit footer on cover page (page 1)
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

def build_pdf(question, answer, trace):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ------------------ PAGE 1: COVER PAGE ------------------
    pdf.add_page()
    
    # Add decorative border line
    pdf.set_draw_color(115, 25, 44) # Burgundy
    pdf.set_line_width(1)
    pdf.rect(10, 10, 190, 277)
    
    # University Logo centered
    # Logo dimensions: 70 width, aspect ratio 150/400 = 0.375 -> height = 26.25
    logo_w = 70
    logo_h = 26
    logo_x = (210 - logo_w) / 2
    pdf.image(LOGO_PATH, x=logo_x, y=60, w=logo_w, h=logo_h)
    
    # Lab Title
    pdf.set_y(115)
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 10, 'LABORATORY RECORD', border=0, align='C')
    pdf.ln(12)
    
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, 'Experiment 4: SQL Agent with Tool Use', border=0, align='C')
    pdf.ln(8)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '(ReAct Framework)', border=0, align='C')
    
    # Student Details
    pdf.set_y(210)
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 8, 'Submitted By:', border=0, align='C')
    pdf.ln(8)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, 'D. megha vardhan', border=0, align='C')
    pdf.ln(8)
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 8, 'Roll No: 2311cs040180', border=0, align='C')
    
    pdf.set_y(250)
    pdf.set_font('helvetica', 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, 'Department of Computer Science & Engineering', border=0, align='C')
    pdf.ln(6)
    pdf.cell(0, 6, 'Malla Reddy University, Hyderabad', border=0, align='C')

    # ------------------ PAGE 2: OBJECTIVE & SCHEMA ------------------
    pdf.add_page()
    
    # Section 1: Objective
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 10, '1. Objective & System Architecture', border='B')
    pdf.ln(12)
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, 
        "The objective of this experiment is to design and implement a ReAct-based (Reasoning and Acting) "
        "SQL agent. Unlike static Text-to-SQL workflows, a ReAct agent operates in iterative cycles: "
        "thinking about the task, executing dynamic database tools to gather schema or row insights, "
        "observing tool output, and then formulating the next action. This allows the model to "
        "troubleshoot query errors, list tables, and analyze table schemas dynamically before running "
        "a final safe read-only SQL query."
    )
    pdf.ln(6)
    
    # Section 2: ReAct Loop Diagram
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 8, 'The ReAct Reasoning & Execution Cycle')
    pdf.ln(8)
    
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_font('courier', 'B', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, ' [User Query] -> Thought -> Action -> Observation -> [Loop] -> Final Answer', border=1, fill=True, align='C')
    pdf.ln(12)
    
    # Section 3: Database Tools
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 10, '2. E-Commerce Database Schema & Tools', border='B')
    pdf.ln(12)
    
    pdf.set_font('helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, 
        "The agent has access to a local SQLite database (ecommerce.db) containing three tables:\n"
        "1. customers: Stores customer names, emails, countries, and signup dates.\n"
        "2. products: Stores product inventory, pricing, and category classifications.\n"
        "3. orders: Transaction log mapping customer purchases to product quantities.\n\n"
        "The agent utilizes three specific database tools:\n"
        " - list_tables: No arguments. Returns all table names in the database.\n"
        " - get_schema: Argument: table_name. Returns table DDL details.\n"
        " - execute_query: Argument: SQL query. Runs read-only SELECT queries."
    )
    
    # ------------------ PAGE 3: EXECUTION TRACE ------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 10, '3. Agent Execution Trace', border='B')
    pdf.ln(12)
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"User Question: '{question}'")
    pdf.ln(8)
    
    # Draw execution trace inside a console-style box
    pdf.set_fill_color(240, 240, 242)
    pdf.set_draw_color(115, 25, 44)
    pdf.set_font('courier', '', 8)
    pdf.set_text_color(30, 30, 30)
    
    # Let's concatenate trace steps nicely
    trace_text = ""
    for idx, step in enumerate(trace):
        # Format step blocks
        if step.startswith("Question:"):
            continue
        trace_text += f"{step}\n"
        trace_text += "-" * 60 + "\n"
        
    pdf.multi_cell(0, 4.5, trace_text.strip(), border=1, fill=True)
    pdf.ln(8)
    
    # Section 4: Final Output Callout
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(115, 25, 44)
    pdf.cell(0, 8, 'Synthesized Final Answer')
    pdf.ln(8)
    
    pdf.set_fill_color(115, 25, 44) # Burgundy fill
    pdf.set_text_color(255, 255, 255) # White text
    pdf.set_font('helvetica', 'B', 11)
    pdf.multi_cell(0, 8, f" {answer}", border=0, fill=True)
    
    # Save the file
    pdf.output(PDF_PATH)
    print(f"PDF report successfully saved to: {PDF_PATH}")

if __name__ == "__main__":
    prepare_logo()
    
    # Execute the ReAct agent on our test question to get fresh execution trace
    question = "What is the total revenue of products bought by Alice Smith?"
    answer, trace = run_agent(question)
    
    # Generate the PDF report
    build_pdf(question, answer, trace)
