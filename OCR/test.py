from ocr_service import TESSERACT_AVAILABLE
import pytesseract

if not TESSERACT_AVAILABLE:
    raise SystemExit(
        "Tesseract was not found. Install it or set TESSERACT_CMD to tesseract.exe."
    )

print(pytesseract.get_tesseract_version())
