# OCR API

FastAPI-based OCR service for extracting text and structured fields from images and PDFs. It supports common document types such as PAN cards, Aadhaar cards, LOS portal screenshots, invoices, salary slips, bank statements, and loan documents.

## Features

- Upload image or PDF documents.
- Extract cleaned OCR text.
- Detect structured fields such as PAN number, name, DOB, phone, email, dates, amounts, loan login number, branch, stage, and document type.
- Supports image OCR through Tesseract.
- Supports PDF text extraction through `pdfplumber`.
- Supports scanned PDF rendering through Poppler / `pdf2image`.
- Optional Claude Vision fallback when `ANTHROPIC_API_KEY` is configured.
- Stores uploaded document metadata in `documents.json`.

## Requirements

- Python 3.10+
- Tesseract OCR
- Poppler for scanned PDF support

Configured local paths:

```text
Tesseract:
C:\Users\M3084\AppData\Local\Programs\Tesseract-OCR\tesseract.exe

Poppler:
C:\Users\M3084\AppData\Local\Programs\poppler-26.02.0\Library\bin
```

The app also supports environment variables:

```powershell
$env:TESSERACT_CMD="C:\Users\M3084\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
$env:POPPLER_PATH="C:\Users\M3084\AppData\Local\Programs\poppler-26.02.0\Library\bin"
```

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## API

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "OCR API Running"
}
```

### Upload Document

```http
POST /upload
```

Form field:

```text
file: image or PDF file
```

Allowed extensions:

```text
.jpg, .jpeg, .png, .bmp, .tiff, .tif, .pdf
```

Example response:

```json
{
  "status": "success",
  "document_id": "uuid",
  "file_name": "PAN.pdf",
  "confidence": 95,
  "ocr_engine": "tesseract",
  "fields": {
    "document_type": "PAN Card",
    "pan_number": "ABCDE1234F",
    "name": "NAME",
    "dob": "01/01/1990"
  },
  "raw_text": "Extracted text..."
}
```

### Get Document

```http
GET /document/{document_id}
```

Returns the stored record from `documents.json`.

## Project Files

```text
main.py           FastAPI app and upload endpoints
ocr_service.py    OCR pipeline, PDF rendering, text cleanup, field extraction
database.py       Optional MongoDB connection helper
documents.json    JSON document store
uploads/          Uploaded files
test.py           Tesseract configuration check
requirements.txt  Python dependencies
```

## Troubleshooting

### Poppler Not Found

If PDF upload fails with a Poppler error, confirm these files exist:

```powershell
Test-Path "C:\Users\M3084\AppData\Local\Programs\poppler-26.02.0\Library\bin\pdfinfo.exe"
Test-Path "C:\Users\M3084\AppData\Local\Programs\poppler-26.02.0\Library\bin\pdftoppm.exe"
```

Both should return `True`.

### Tesseract Not Found

Run:

```powershell
python test.py
```

It should print the installed Tesseract version.

If it fails, set:

```powershell
$env:TESSERACT_CMD="C:\Users\M3084\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
```

### PDF Has No Text

Some PDFs are scanned images, not digital text. For those, the app renders PDF pages as images and runs Tesseract OCR. Poppler or PyMuPDF must be available for scanned PDFs.

### Low OCR Accuracy

OCR accuracy depends on image quality. For best results:

- Use clear, high-resolution images.
- Avoid tilted or blurry photos.
- Crop the document tightly.
- Avoid glare, shadows, and noisy backgrounds.

## Optional Claude Vision Fallback

Set `ANTHROPIC_API_KEY` to enable Claude Vision fallback for low-confidence OCR:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key"
```

Without this key, the app uses Tesseract only.
