from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
import os
import logging
import tempfile
import uuid
import datetime
from src.pipeline import BISRAGEngine

# Import document parsers (will be installed by user)
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BIS SmartStandards API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    logger.info("Initializing BIS RAG Engine...")
    engine = BISRAGEngine()
    # Ensure index is built on startup
    engine.build_index()
    logger.info("Engine ready.")

class PredictRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    target_lang: str

def format_response(response, latency, metadata=None):
    standards = []
    for s in response.get("standards", []):
        std_id = s.get("standard", "Unknown")
        reason_text = s.get("reason", "")
        
        # Try to extract title from reason (format: "[Title] Applicable for...")
        title = "BIS Standard"
        if reason_text.startswith("[") and "]" in reason_text:
            title = reason_text[1:reason_text.index("]")]
            reason_text = reason_text[reason_text.index("]")+1:].strip()
            
        standards.append({
            "standard": std_id,
            "title": title,
            "confidence": s.get("confidence", 0.8),
            "reason": reason_text
        })
        
    return {
        "recommended_standards": standards,
        "response_time": latency,
        "metadata": metadata or {}
    }

def extract_metadata_from_text(text: str) -> dict:
    import re
    metadata = {
        "Product Name": "Unspecified / Analyzed from Document",
        "Product Category": "Building Materials",
        "Manufacturer Name": "N/A",
        "Usage / Application": "Refer to Document",
        "Manufacturing Process": "Standard Processing"
    }
    
    invalid_keywords = ["Describe", "e.g.", "Please", "Where and how", "Refer to"]
    
    def extract_field(field_pattern, text_content):
        pattern_same_line = rf"{field_pattern}[\s_:]+([A-Za-z0-9][^\n]+)"
        match = re.search(pattern_same_line, text_content, re.IGNORECASE)
        if match:
            val = match.group(1).replace('_', '').strip()
            if val and not val.endswith(':'):
                if not any(k in val for k in invalid_keywords):
                    return val
        
        pattern_next_line = rf"{field_pattern}[^\n]*\n+([A-Za-z0-9][^\n]+)"
        match = re.search(pattern_next_line, text_content, re.IGNORECASE)
        if match:
            val = match.group(1).replace('_', '').strip()
            if val and not val.endswith(':'):
                if not any(k in val for k in invalid_keywords):
                    return val
        return None

    pn = extract_field(r"Product Name", text)
    if pn: metadata["Product Name"] = pn[:100]
    
    cat = extract_field(r"(?:Material Category|Product Category)", text)
    if cat: metadata["Product Category"] = cat[:100]
    
    use = extract_field(r"(?:Intended Usage|Application|Usage \/ Application|Description|PRODUCT DESCRIPTION)", text)
    if use: metadata["Usage / Application"] = use[:150]
        
    mfg = extract_field(r"(?:Company Name|Manufacturer Name|Manufacturer)", text)
    if mfg: metadata["Manufacturer Name"] = mfg[:100]
    
    return metadata

@app.post("/predict")
async def predict(req: PredictRequest):
    global engine
    if not engine:
        raise HTTPException(status_code=503, detail="Engine is still initializing.")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        start_time = time.time()
        metadata = extract_metadata_from_text(req.text)
        response = engine.query(req.text)
        latency = round(time.time() - start_time, 2)
        return format_response(response, latency, metadata=metadata)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text(req: TranslateRequest):
    import urllib.request
    import urllib.parse
    import json
    
    if req.target_lang == "en":
        return {"translated_text": req.text}
        
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={req.target_lang}&dt=t&q={urllib.parse.quote(req.text)}"
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode())
            translated_text = "".join([sentence[0] for sentence in data[0] if sentence[0]])
            return {"translated_text": translated_text}
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return {"translated_text": req.text}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global engine
    if not engine:
        raise HTTPException(status_code=503, detail="Engine is still initializing.")
    
    # Check file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["pdf", "docx"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX are supported.")
        
    # Check file size (Read into memory to check size)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")
        
    # Save temporarily to extract text
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp:
        temp.write(contents)
        temp_path = temp.name
        
    try:
        if ext == "pdf":
            if not fitz:
                raise HTTPException(status_code=500, detail="PyMuPDF not installed")
            doc = fitz.open(temp_path)
            for page in doc:
                text += page.get_text("text", sort=True) + "\n"
            doc.close()
        elif ext == "docx":
            if not docx:
                raise HTTPException(status_code=500, detail="python-docx not installed")
            doc = docx.Document(temp_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass
        raise HTTPException(status_code=400, detail="Unable to read the document. Please upload a valid PDF or Word file.")
        
    try:
        os.unlink(temp_path)
    except:
        pass
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the document.")
        
    # Extract simple metadata safely
    metadata = extract_metadata_from_text(text)

    # Send combined text to prediction engine
    try:
        start_time = time.time()
        response = engine.query(text[:2000]) # Pass first 2000 chars to avoid prompt overflow
        latency = round(time.time() - start_time, 2)
        return format_response(response, latency, metadata=metadata)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/template/pdf")
async def get_template_pdf():
    file_path = "frontend/assets/template.pdf"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="BIS_Product_Information_Form.pdf", media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Template not found")

@app.get("/api/template/docx")
async def get_template_docx():
    file_path = "frontend/assets/template.docx"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="BIS_Product_Information_Form.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    raise HTTPException(status_code=404, detail="Template not found")

class ReportRequest(BaseModel):
    query_text: str = "Uploaded Document Analysis"
    standards: list
    metadata: dict = None

def generate_report_pdf(data: ReportRequest, filepath: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise HTTPException(status_code=500, detail="ReportLab not installed")

    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles['Heading1'],
        fontName="Helvetica-Bold",
        fontSize=18,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    h2_style = ParagraphStyle(
        name="H2Style",
        parent=styles['Heading2'],
        fontName="Helvetica-Bold",
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )
    
    normal_style = styles['Normal']
    normal_style.fontName = "Helvetica"
    normal_style.fontSize = 11
    normal_style.spaceAfter = 6
    
    bold_style = ParagraphStyle(
        name="BoldStyle",
        parent=normal_style,
        fontName="Helvetica-Bold"
    )

    Story = []
    
    # Title
    Story.append(Paragraph("BIS SmartStandards — Compliance Recommendation Report", title_style))
    Story.append(Spacer(1, 10))
    
    # SECTION 1
    Story.append(Paragraph("SECTION 1 — Report Information", h2_style))
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Story.append(Paragraph(f"<b>Report ID:</b> {report_id}", normal_style))
    Story.append(Paragraph(f"<b>Date and Time:</b> {timestamp}", normal_style))
    Story.append(Paragraph("<b>Generated By:</b> BIS SmartStandards AI Engine", normal_style))
    Story.append(Spacer(1, 10))
    
    # SECTION 2
    Story.append(Paragraph("SECTION 2 — Recommended BIS Standards", h2_style))
    for s in data.standards:
        std_id = s.get("standard", "N/A")
        title = s.get("title", "N/A")
        conf = int(s.get("confidence", 0.8) * 100)
        reason = s.get("reason", "N/A")
        
        Story.append(Paragraph(f"<b>{std_id}</b>", bold_style))
        Story.append(Paragraph(f"{title}", normal_style))
        Story.append(Paragraph(f"<b>Match Score:</b> {conf}%", normal_style))
        Story.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
        Story.append(Spacer(1, 10))
    
    # SECTION 3
    Story.append(Paragraph("SECTION 3 — Key Compliance Areas", h2_style))
    Story.append(Paragraph("• Quality Requirements", normal_style))
    Story.append(Paragraph("• Testing Methods", normal_style))
    Story.append(Paragraph("• Safety Standards", normal_style))
    Story.append(Paragraph("• Material Specifications", normal_style))
    Story.append(Spacer(1, 10))
    
    # SECTION 4
    Story.append(Paragraph("SECTION 4 — Matching Explanation", h2_style))
    Story.append(Paragraph("The selected standards were identified by matching the semantic patterns of the uploaded product description with the official BIS SP 21 corpus. The AI specifically correlated material composition, usage intent, and manufacturing methods to retrieve these exact matches.", normal_style))
    Story.append(Spacer(1, 10))
    
    # SECTION 5
    Story.append(Paragraph("SECTION 5 — Additional Related Standards", h2_style))
    Story.append(Paragraph("Please consult the BIS portal for secondary material handling guidelines.", normal_style))
    Story.append(Spacer(1, 10))
    
    # SECTION 6
    Story.append(Paragraph("SECTION 6 — Disclaimer", h2_style))
    Story.append(Paragraph("<i>This report is generated using an AI-based recommendation system. Users should verify standards with official Bureau of Indian Standards documentation before implementation.</i>", normal_style))
    
    doc.build(Story)

@app.post("/generate-report")
async def generate_report_endpoint(req: ReportRequest):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"BIS_Report_{timestamp}.pdf"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        filepath = tmp.name
        
    generate_report_pdf(req, filepath)
    
    return FileResponse(filepath, filename=filename, media_type="application/pdf")

# Mount the frontend directory so FastAPI serves the HTML/CSS/JS
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Automatically copy generated bot icon and slideshow images for frontend
import shutil
import os
try:
    images_to_copy = [
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\chatbot_robot_icon_1777315161089.png", "frontend/assets/bot_icon.png"),
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\bis_lab_1777317046264.png", "frontend/assets/bis_lab.png"),
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\bis_construction_1777317061988.png", "frontend/assets/bis_construction.png"),
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\bis_manufacturing_1777317076032.png", "frontend/assets/bis_manufacturing.png"),
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\bis_engineers_1777317089734.png", "frontend/assets/bis_engineers.png"),
        (r"C:\Users\HP\.gemini\antigravity\brain\a13edea9-784a-4757-93e5-411d6ea13a1a\bis_quality_1777317108350.png", "frontend/assets/bis_quality.png"),
    ]
    for src, dst in images_to_copy:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
except Exception as e:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
