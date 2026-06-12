import os
import uuid
import json
from json import JSONDecodeError
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ocr_service import extract_text, ALLOWED_EXTENSIONS

app = FastAPI(title="OCR API", version="1.0")

UPLOAD_FOLDER = "uploads"
DATA_FILE = "documents.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "OCR API Running"}


# ---------------- HELPERS ---------------- #

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (JSONDecodeError, OSError):
            return []
    return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------------- ENDPOINTS ---------------- #

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        )

    file_id = str(uuid.uuid4())
    safe_filename = Path(file.filename).name
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{safe_filename}")

    # Save uploaded file to disk
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print("Saved file:", file_path)

    # Run OCR
    result = extract_text(file_path)
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    # Build document record
    document = {
        "document_id":  file_id,
        "file_name":    safe_filename,
        "file_path":    file_path,
        "raw_text":     result["raw_text"],
        "pan_number":   result.get("pan_number"),
        "name":         result.get("name"),
        "dob":          result.get("dob"),
        "fields":       result.get("fields", {}),
        "confidence":   result["confidence"],
        "ocr_engine":   result.get("ocr_engine", "tesseract"),
    }

    data = load_data()
    data.append(document)
    save_data(data)

    return {
        "status":        "success",
        "document_id":   file_id,
        "file_name":     safe_filename,
        "confidence":    result["confidence"],
        "ocr_engine":    result.get("ocr_engine", "tesseract"),
        # structured fields (document_type, name, pan, dob, phone, referral_code …)
        "fields":        result.get("fields", {}),
        # full cleaned text
        "raw_text":      result["raw_text"],
    }


@app.get("/document/{document_id}")
def get_document(document_id: str):
    data = load_data()
    for doc in data:
        if doc["document_id"] == document_id:
            return doc
    raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
