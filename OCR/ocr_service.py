import cv2
import pytesseract
import os
import re
import shutil
import sys
import numpy as np

# ---------------- CROSS-PLATFORM TESSERACT PATH ---------------- #

def configure_tesseract():
    """
    Configure pytesseract from an environment variable, PATH, or common
    Windows install locations. Returns True when an executable is available.
    """
    candidates = []
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        candidates.append(path_cmd)

    if sys.platform == "win32":
        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\M3084\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True

    return False


TESSERACT_AVAILABLE = configure_tesseract()


def find_poppler_path():
    candidates = []
    env_path = os.environ.get("POPPLER_PATH")
    if env_path:
        candidates.append(env_path)

    if sys.platform == "win32":
        candidates.extend([
            r"C:\Users\M3084\AppData\Local\Programs\poppler-26.02.0\Library\bin",
            r"C:\Program Files\poppler\Library\bin",
            r"C:\Program Files (x86)\poppler\Library\bin",
        ])

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            pdfinfo = os.path.join(candidate, "pdfinfo.exe" if sys.platform == "win32" else "pdfinfo")
            pdftoppm = os.path.join(candidate, "pdftoppm.exe" if sys.platform == "win32" else "pdftoppm")
            if os.path.exists(pdfinfo) and os.path.exists(pdftoppm):
                return candidate

    return None


POPPLER_PATH = find_poppler_path()


# ================================================================
# PDF SUPPORT  — 3-strategy fallback chain
# Strategy 1: pdfplumber  — native text extraction (best for digital PDFs)
# Strategy 2: pdf2image + Tesseract — for scanned PDFs
# Strategy 3: Claude Vision page-by-page — last resort
# ================================================================

def is_pdf(file_path):
    return file_path.lower().endswith(".pdf")


def extract_pdf_native_text(file_path):
    """
    Strategy 1: extract embedded text directly via pdfplumber.
    Returns (text, success). Works only for digital/typed PDFs, not scans.
    """
    try:
        import pdfplumber
        all_text = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    header = f"--- Page {i+1} ---\n" if len(pdf.pages) > 1 else ""
                    all_text.append(header + page_text.strip())
        text = "\n\n".join(all_text)
        # Consider it a success only if we got meaningful text (>20 chars)
        return text, len(text.strip()) > 20
    except ImportError:
        print("pdfplumber not installed — skipping native text extraction")
        return "", False
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        return "", False


def pdf_to_images(file_path, dpi=300):
    """
    Render PDF pages as images for OCR.
    PyMuPDF is tried first because it does not require Poppler. pdf2image is
    kept as a fallback for environments that already have Poppler installed.
    """
    pymupdf_error = None
    try:
        import fitz

        images = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(file_path) as pdf:
            for page in pdf:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 1:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_bgr)
        if images:
            return images, None
        pymupdf_error = "PyMuPDF returned no pages"
    except ImportError:
        pymupdf_error = "PyMuPDF not installed. Run: pip install PyMuPDF"
    except Exception as e:
        pymupdf_error = f"PyMuPDF error: {e}"

    try:
        from pdf2image import convert_from_path
        kwargs = {"dpi": dpi}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
        pages = convert_from_path(file_path, **kwargs)
        images = []
        for page in pages:
            img_array = np.array(page)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
        if not images:
            raise RuntimeError("pdf2image returned no pages")
        return images, None
    except ImportError:
        return [], f"{pymupdf_error}; pdf2image not installed. Run: pip install pdf2image"
    except Exception as e:
        msg = str(e)
        if "poppler" in msg.lower() or "pdftoppm" in msg.lower() or "pdfinfo" in msg.lower():
            return [], (
                f"{pymupdf_error}; Poppler not found. Install PyMuPDF with "
                "`pip install PyMuPDF`, or install Poppler:\n"
                "  Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
                "  Mac:     brew install poppler\n"
                "  Linux:   apt install poppler-utils"
            )
        return [], f"{pymupdf_error}; pdf2image error: {msg}"


def pdf_pages_via_claude(file_path):
    """
    Strategy 3: send each PDF page image to Claude Vision.
    BUG FIX: was blocked by 'not is_pdf()' check — now explicitly supported.
    """
    try:
        import base64, json, urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "", 0

        pages, render_error = pdf_to_images(file_path, dpi=200)
        if not pages:
            print(f"Claude Vision PDF render failed: {render_error}")
            return "", 0

        all_texts = []

        for i, page in enumerate(pages):
            ok, encoded = cv2.imencode(".jpg", page, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                all_texts.append(f"--- Page {i+1}: image encoding failed ---")
                continue
            b64 = base64.standard_b64encode(encoded.tobytes()).decode("utf-8")

            payload = json.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL visible text from this document page exactly as printed. "
                                "Include every word, number, label, date, and field value. "
                                "Output only the extracted text with no commentary."
                            )
                        }
                    ]
                }]
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())
                    page_text = result["content"][0]["text"].strip()
                    header = f"--- Page {i+1} ---\n" if len(pages) > 1 else ""
                    all_texts.append(header + page_text)
            except Exception as e:
                print(f"Claude Vision failed for page {i+1}: {e}")
                all_texts.append(f"--- Page {i+1}: extraction failed ---")

        return "\n\n".join(all_texts), 90

    except ImportError:
        return "", 0
    except Exception as e:
        print(f"pdf_pages_via_claude failed: {e}")
        return "", 0


# ================================================================
# TEXT CLEANING
# ================================================================

# Characters that are pure OCR noise — icons, bullets, box-drawing glyphs
_NOISE_CHARS = re.compile(
    r"[\u00a9\u00ae\u2122"          # © ® ™
    r"\u2018\u2019\u201c\u201d"     # curly quotes → keep content, strip later
    r"\u2022\u2023\u25aa\u25cf"     # bullet variants
    r"\u2500-\u257f"                # box-drawing
    r"\u2580-\u259f"                # block elements
    r"\uf000-\ufffd"                # private use / specials
    r"\x00-\x08\x0b\x0c\x0e-\x1f"  # control chars (keep \t \n \r)
    r"]"
)

# Icon-like single characters that OCR misreads from UI elements
_ICON_TOKENS = re.compile(
    r"(?<!\w)"                       # not preceded by word char
    r"[&@#\*\^\~\|\\\/<>]"          # punctuation used as icon placeholders
    r"(?!\w)"                        # not followed by word char
)

# Collapse 3+ consecutive blank lines into 2
_EXCESS_BLANK = re.compile(r"\n{3,}")

# Trailing whitespace on each line
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_text(raw: str) -> str:
    """
    Remove OCR noise while preserving real content.
    Handles: stray symbols, icon artifacts, excess blank lines,
    curly quotes → straight, non-breaking spaces → regular spaces.
    """
    if not raw:
        return ""

    text = raw

    # 1. Normalise whitespace variants
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")   # non-breaking space → space
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # 2. Strip noise characters
    text = _NOISE_CHARS.sub("", text)

    # 3. Remove lone icon-like punctuation tokens
    text = _ICON_TOKENS.sub("", text)

    # 4. Clean trailing whitespace per line
    text = _TRAILING_WS.sub("", text)

    # 5. Collapse excessive blank lines
    text = _EXCESS_BLANK.sub("\n\n", text)

    # 6. Strip leading/trailing whitespace from whole document
    text = text.strip()

    return text


# ================================================================
# STRUCTURED FIELD EXTRACTION  — handles all document types
# ================================================================

# ----------------------------------------------------------------
# 1. Document type detection
# ----------------------------------------------------------------

# Rules ordered from most-specific to most-generic.
# Each entry: (required_keywords, any_of_keywords, label)
# required_keywords  — ALL must be present
# any_of_keywords    — at least ONE must be present (empty = skip check)
_DOC_TYPE_RULES = [
    # LOS / Loan Origination System portal screenshots
    (["los"],                   ["stage", "loan-details", "loginno", "micro lap", "dms", "collateral"], "LOS Portal Screenshot"),
    # Identity documents
    (["permanent account number"], [],                                                                   "PAN Card"),
    (["income tax"],              ["pan", "permanent"],                                                  "PAN Card"),
    (["govt of india"],           ["income", "department", "father", "birth"],                          "PAN Card"),
    ([],                          ["aadhaar", "aadhar", "unique identification authority"],              "Aadhaar Card"),
    ([],                          ["passport no", "republic of india", "place of issue"],               "Passport"),
    ([],                          ["driving licence", "driving license", "dl no"],                      "Driving Licence"),
    ([],                          ["voter id", "election commission", "epic no"],                       "Voter ID"),
    # Financial documents
    ([],                          ["salary slip", "pay slip", "gross salary", "net pay"],               "Salary Slip"),
    ([],                          ["bank statement", "opening balance", "closing balance"],              "Bank Statement"),
    ([],                          ["form 16", "tds certificate", "tax deducted at source"],             "Form 16 / TDS Certificate"),
    (["invoice"],                 ["bill to", "total amount", "gst", "hsn"],                            "Invoice"),
    ([],                          ["sanction letter", "loan sanction"],                                 "Loan Sanction Letter"),
    ([],                          ["loan agreement", "borrower", "lender", "repayment schedule"],       "Loan Agreement"),
    # Letters / comms
    (["thank you"],               ["referral code", "representatives will contact", "loan services"],   "Loan Acknowledgement Letter"),
    ([],                          ["noc", "no objection certificate"],                                  "NOC"),
    (["dear"],                    ["loan", "application", "regarding"],                                  "Loan Letter"),
]


def detect_document_type(text: str) -> str:
    t = text.lower()
    for required, any_of, label in _DOC_TYPE_RULES:
        if required and not all(k in t for k in required):
            continue
        if any_of and not any(k in t for k in any_of):
            continue
        return label
    return "Document"


# ----------------------------------------------------------------
# 2. Field detectors — generic
# ----------------------------------------------------------------

def _first(pattern, text, flags=0, group=1):
    """Return first match group or None."""
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _valid_person_name(value: str) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if "http" in lowered or "://" in lowered or "tabname" in lowered:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    words = [w for w in re.split(r"\s+", value.strip()) if w]
    return 1 <= len(words) <= 5 and all(re.fullmatch(r"[A-Za-z][A-Za-z.']*", w) for w in words)


def detect_pan_number(text: str):
    m = re.findall(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text.upper())
    if not m:
        m = re.findall(r"[A-Z]{5}[0-9]{4}[A-Z]", text.upper())
    return m[0] if m else None


def detect_name(text: str):
    # Pattern 1: greeting  "Dear ANIL TIWARI,"
    m = re.search(r"\bDear\s+([A-Z][A-Za-z\s\.]{2,60}?)[\s,\n]", text)
    if m:
        name = m.group(1).strip().rstrip(",")
        if _valid_person_name(name):
            return name
    # Pattern 2: "Name: ..." or "Name\n..."
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for i, line in enumerate(lines):
        if re.fullmatch(r"name", line, re.IGNORECASE):
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if _valid_person_name(candidate):
                    return candidate
        m = re.match(r"^\s*(?:full\s+)?name\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if _valid_person_name(candidate):
                return candidate
    return None


def detect_dob(text: str):
    labeled = _first(
        r"(?:date\s+of\s+birth|dob|birth\s+date)[\s:\-]*"
        r"(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{1,2}\s+\w+\s+\d{4}|"
        r"\w+\s+\d{1,2},?\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if labeled:
        return labeled
    return _first(
        r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{1,2}\s+\w+\s+\d{4}|"
        r"\w+\s+\d{1,2},?\s+\d{4})\b", text
    )


def detect_labeled_dob(text: str):
    return _first(
        r"(?:date\s+of\s+birth|dob|birth\s+date)[\s:\-]*"
        r"(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{1,2}\s+\w+\s+\d{4}|"
        r"\w+\s+\d{1,2},?\s+\d{4})",
        text,
        re.IGNORECASE,
    )


def detect_phone(text: str):
    m = re.search(r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}", text)
    return m.group(0).strip() if m else None


def detect_referral_code(text: str):
    return _first(
        r"(?:referral\s*code|ref(?:erence)?\s*(?:code|no)?|promo\s*code)"
        r"[\s:\-]*([A-Za-z0-9]{4,20})",
        text, re.IGNORECASE
    )


def detect_email(text: str):
    return _first(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text, group=0)


def detect_date(text: str, label_hint: str = ""):
    """Generic date finder — looks near an optional label."""
    pattern = (
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
        r"\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b"
    )
    if label_hint:
        scoped = re.search(
            label_hint + r"[\s:\-]*" + pattern, text, re.IGNORECASE
        )
        if scoped:
            return scoped.group(1).strip()
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def detect_amount(text: str):
    """Find the largest currency amount — useful for invoices/loan docs."""
    amounts = re.findall(
        r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE
    )
    if not amounts:
        return None
    # return the largest value found
    def parse(s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return 0
    return max(amounts, key=parse)


# ----------------------------------------------------------------
# 3. Document-type-specific extractors
# ----------------------------------------------------------------

def _extract_los_fields(text: str) -> dict:
    """
    LOS portal screenshot — extract loan application metadata.
    Example line: MW12721210 | Test Case - Assessed Income Prog | 07-HEAD OFFICE | Micro Lap | Nov 18, 2025
    """
    fields = {}

    # Login / Application number  (MW followed by digits)
    m = re.search(r"\b(MW\d{6,12})\b", text, re.IGNORECASE)
    if m: fields["login_no"] = m.group(1).upper()

    # Stage
    m = re.search(r"Stage\s*[-–:]\s*([A-Za-z -]+)", text, re.IGNORECASE)
    if m: fields["stage"] = m.group(1).strip()

    # Branch / office
    m = re.search(r"\b(\d{2}-[A-Z\s]+(?:OFFICE|BRANCH|HO|RO|ZO)[A-Z\s]*)", text)
    if m: fields["branch"] = m.group(1).strip()

    # Loan product / program  (line after login no, before branch)
    prog = re.search(
        r"MW\d+\s*\|\s*([^|]+?)\s*\|", text, re.IGNORECASE
    )
    if prog: fields["program"] = prog.group(1).strip()

    # Loan type (Micro Lap, Home Loan, etc.)
    loan_type = re.search(
        r"\|\s*(Micro\s+Lap|Home\s+Loan|Personal\s+Loan|Business\s+Loan|"
        r"Gold\s+Loan|LAP|SME|MSME|Auto\s+Loan)\s*[|\n]",
        text, re.IGNORECASE
    )
    if loan_type: fields["loan_type"] = loan_type.group(1).strip()

    # Date
    d = detect_date(text)
    if d: fields["date"] = d

    # Active tab / section
    m = re.search(
        r"\b(Basic Details|Income|DMS|Collateral|Legal|PD|Credit)\b", text
    )
    if m: fields["active_section"] = m.group(1)

    # Logged-in user  (Employee ID pattern like M3084)
    m = re.search(r"\b([A-Z]\d{4,6})\b", text)
    if m: fields["employee_id"] = m.group(1)

    # Logged-in user name — appears before employee ID in header
    user = re.search(
        r"([A-Z][A-Z\s]{3,30})\n([A-Z]\d{4,6})", text
    )
    if user: fields["user_name"] = user.group(1).strip()

    return fields


def _extract_pan_fields(text: str) -> dict:
    fields = {}
    pan = detect_pan_number(text)
    if pan: fields["pan_number"] = pan
    name = detect_name(text)
    if name: fields["name"] = name
    dob = detect_labeled_dob(text)
    if dob: fields["dob"] = dob
    # Father's name — line after "Father's Name" label
    father = _first(
        r"father['\u2019s]*\s*name[\s:\n]+([A-Z][A-Za-z\s\.]{2,50})", text, re.IGNORECASE
    )
    if father: fields["father_name"] = father
    return fields


def _extract_aadhaar_fields(text: str) -> dict:
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    dob = detect_labeled_dob(text)
    if dob: fields["dob"] = dob
    # Aadhaar number — 4-4-4 digit groups
    uid = re.search(r"\b(\d{4}\s\d{4}\s\d{4})\b", text)
    if uid: fields["aadhaar_number"] = uid.group(1)
    phone = detect_phone(text)
    if phone: fields["phone"] = phone
    return fields


def _extract_salary_fields(text: str) -> dict:
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    # Month/Year of salary
    period = _first(r"(?:salary\s+for|pay\s+period|month)[:\s]+([A-Za-z]+\s+\d{4})", text, re.IGNORECASE)
    if period: fields["pay_period"] = period
    # Gross / Net salary
    gross = _first(r"gross\s+(?:salary|pay|earnings)[:\s]+([\d,\.]+)", text, re.IGNORECASE)
    if gross: fields["gross_salary"] = gross
    net = _first(r"net\s+(?:salary|pay)[:\s]+([\d,\.]+)", text, re.IGNORECASE)
    if net: fields["net_salary"] = net
    emp_id = _first(r"(?:employee\s+id|emp\.?\s*id|staff\s+id)[:\s]+([A-Za-z0-9\-]+)", text, re.IGNORECASE)
    if emp_id: fields["employee_id"] = emp_id
    return fields


def _extract_bank_statement_fields(text: str) -> dict:
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    acc = _first(r"(?:account\s+(?:no|number)|a/?c\s+no)[:\s.]+([0-9Xx]{6,20})", text, re.IGNORECASE)
    if acc: fields["account_number"] = acc
    ifsc = _first(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", text)
    if ifsc: fields["ifsc_code"] = ifsc
    bank = _first(r"(?:bank\s+name|bank)[:\s]+([A-Za-z\s]+(?:Bank|Financial)[A-Za-z\s]*)", text, re.IGNORECASE)
    if bank: fields["bank_name"] = bank.strip()
    return fields


def _extract_invoice_fields(text: str) -> dict:
    fields = {}
    inv_no = _first(r"(?:invoice\s+(?:no|number)|inv\.?\s*no)[:\s#]+([A-Za-z0-9\-\/]+)", text, re.IGNORECASE)
    if inv_no: fields["invoice_number"] = inv_no
    d = detect_date(text, label_hint=r"(?:invoice\s+date|date)")
    if d: fields["invoice_date"] = d
    amt = detect_amount(text)
    if amt: fields["total_amount"] = amt
    gstin = _first(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b", text)
    if gstin: fields["gstin"] = gstin
    return fields


def _extract_acknowledgement_fields(text: str) -> dict:
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    phone = detect_phone(text)
    if phone: fields["phone"] = phone
    ref = detect_referral_code(text)
    if ref: fields["referral_code"] = ref
    email = detect_email(text)
    if email: fields["email"] = email
    return fields


def _extract_loan_letter_fields(text: str) -> dict:
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    amt = detect_amount(text)
    if amt: fields["loan_amount"] = amt
    phone = detect_phone(text)
    if phone: fields["phone"] = phone
    d = detect_date(text)
    if d: fields["date"] = d
    acc = _first(r"(?:loan\s+account|account\s+no)[:\s.]+([A-Za-z0-9\-\/]+)", text, re.IGNORECASE)
    if acc: fields["loan_account"] = acc
    return fields


def _extract_generic_fields(text: str) -> dict:
    """Fallback — grab whatever common fields are present."""
    fields = {}
    name = detect_name(text)
    if name: fields["name"] = name
    phone = detect_phone(text)
    if phone: fields["phone"] = phone
    email = detect_email(text)
    if email: fields["email"] = email
    d = detect_date(text)
    if d: fields["date"] = d
    pan = detect_pan_number(text)
    if pan: fields["pan_number"] = pan
    dob = detect_labeled_dob(text)
    if dob: fields["dob"] = dob
    amt = detect_amount(text)
    if amt: fields["amount"] = amt
    return fields


# ----------------------------------------------------------------
# 4. Master dispatcher
# ----------------------------------------------------------------

_EXTRACTOR_MAP = {
    "LOS Portal Screenshot":       _extract_los_fields,
    "PAN Card":                    _extract_pan_fields,
    "Aadhaar Card":                _extract_aadhaar_fields,
    "Salary Slip":                 _extract_salary_fields,
    "Bank Statement":              _extract_bank_statement_fields,
    "Invoice":                     _extract_invoice_fields,
    "Loan Acknowledgement Letter": _extract_acknowledgement_fields,
    "Acknowledgement Letter":      _extract_acknowledgement_fields,
    "Loan Agreement":              _extract_loan_letter_fields,
    "Loan Sanction Letter":        _extract_loan_letter_fields,
    "Loan Letter":                 _extract_loan_letter_fields,
}


def extract_all_fields(text: str) -> dict:
    """
    1. Detect document type.
    2. Run the matching specialist extractor.
    3. Fall back to generic extractor for any fields not found.
    Returns only fields that were actually detected.
    """
    doc_type = detect_document_type(text)

    # Run specialist extractor
    extractor = _EXTRACTOR_MAP.get(doc_type, _extract_generic_fields)
    fields = extractor(text)

    # Also run generic pass to catch anything the specialist missed
    generic = _extract_generic_fields(text)
    for k, v in generic.items():
        if k not in fields and v:
            fields[k] = v

    # Always include document_type
    fields["document_type"] = doc_type

    # Remove any None / empty values
    return {k: v for k, v in fields.items() if v}


# ---------------- IMAGE DESKEW ---------------- #

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if coords.shape[0] < 5:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------- DOCUMENT CROP ---------------- #

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 50 or max_height < 50:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def crop_document(image):
    """
    Detect and crop a card/document from a photo. Returns None when no confident
    crop is found, so the caller can safely fall back to the original image.
    """
    h, w = image.shape[:2]
    image_area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:8]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.12 or area > image_area * 0.98:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 4:
            warped = four_point_transform(image, approx.reshape(4, 2))
            if warped is not None:
                return warped

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < w * 0.35 or ch < h * 0.20:
            continue
        pad = max(4, int(min(cw, ch) * 0.03))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        return image[y1:y2, x1:x2]

    return None


# ---------------- IMAGE ENHANCEMENT ---------------- #

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    max_dim = max(h, w)
    if max_dim < 1000:
        scale = min(5.0, max(2.0, 1200 / max_dim))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


# ---------------- THRESHOLD SELECTION ---------------- #

def score_threshold(binary):
    total = binary.size
    white = cv2.countNonZero(binary)
    ratio = white / total
    if ratio > 0.85 or ratio < 0.15:
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(binary), connectivity=8
    )
    h, w = binary.shape
    char_sized = sum(
        1 for i in range(1, num_labels)
        if 5 < stats[i, cv2.CC_STAT_WIDTH] < w // 2
        and 5 < stats[i, cv2.CC_STAT_HEIGHT] < h // 2
    )
    return char_sized


def get_best_image(gray):
    options = {}
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    options["otsu"] = th_otsu
    th_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    options["adaptive_gauss"] = th_gauss
    th_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    options["adaptive_mean"] = th_mean
    scored = {name: score_threshold(img) for name, img in options.items()}
    best_name = max(scored, key=scored.get)
    return options[best_name]


# ---------------- OCR ENGINE ---------------- #

def run_ocr_with_confidence(image):
    if not TESSERACT_AVAILABLE:
        return "", 0

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 3",
        "--oem 3 --psm 11",
    ]
    best_text = ""
    best_conf = -1
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(image, config=cfg, lang="eng",
                                              output_type=pytesseract.Output.DICT)
            confs = []
            for c in data["conf"]:
                try:
                    conf = float(c)
                except (TypeError, ValueError):
                    continue
                if conf >= 0:
                    confs.append(conf)
            if not confs:
                continue
            mean_conf = sum(confs) / len(confs)
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_text = pytesseract.image_to_string(image, config=cfg, lang="eng")
        except pytesseract.TesseractNotFoundError:
            return "", 0
        except Exception:
            continue
    return best_text, max(best_conf, 0)


# ---------------- CLAUDE VISION (IMAGE FILES) ---------------- #

def ocr_via_claude(file_path):
    import base64, json, urllib.request
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, 0
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff"}
    media_type = mime_map.get(ext)
    if not media_type:
        return None, 0
    with open(file_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": (
                    "Extract ALL visible text from this document image exactly as printed. "
                    "Include every word, number, label, and field. "
                    "Then on separate lines:\n"
                    "PAN_NUMBER: <10-char PAN or NONE>\n"
                    "NAME: <full name or NONE>\n"
                    "DOB: <date of birth or NONE>"
                )}
            ]
        }]
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result["content"][0]["text"], 90
    except Exception as e:
        print("Claude Vision fallback failed:", e)
        return None, 0


# ---------------- SINGLE IMAGE OCR PIPELINE ---------------- #

def ocr_single_image(image):
    gray = enhance_image(image)
    processed = get_best_image(gray)
    processed = deskew(processed)
    text, conf = run_ocr_with_confidence(processed)
    if conf < 40:
        gray2 = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed2 = deskew(processed2)
        text2, conf2 = run_ocr_with_confidence(processed2)
        if conf2 > conf:
            text, conf = text2, conf2
    return text, conf


def ocr_image(image):
    variants = [image]
    cropped = crop_document(image)
    if cropped is not None:
        variants.append(cropped)

    results = []
    best_text = ""
    best_conf = -1
    for variant in variants:
        text, conf = ocr_single_image(variant)
        if text.strip():
            results.append((text.strip(), conf))
        text_len = len(text.strip())
        best_len = len(best_text.strip())
        has_more_content = text_len > best_len * 1.2 and conf >= 25
        if conf > best_conf or has_more_content or (conf == best_conf and text_len > best_len):
            best_text = text
            best_conf = conf

    merged = best_text.strip()
    for text, conf in results:
        if text == merged:
            continue
        if conf >= 25 and text not in merged and merged not in text:
            merged = (merged + "\n\n" + text).strip() if merged else text

    return merged, max(best_conf, 0)


# ================================================================
# CORE FUNCTION
# ================================================================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
TESSERACT_CONFIDENCE_THRESHOLD = 50


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return {
            "raw_text": "", "pan_number": None, "name": None, "dob": None,
            "confidence": 0, "error": f"Unsupported file type: {ext}"
        }

    used_claude = False
    all_text = ""
    avg_conf = 0
    tesseract_error = (
        "Tesseract OCR executable was not found. Install Tesseract or set "
        "TESSERACT_CMD to the full path of tesseract.exe."
    )

    # ================================================================
    # PDF PATH — 3-strategy chain
    # ================================================================
    if is_pdf(file_path):

        # Strategy 1: native text (best for digital PDFs)
        native_text, native_ok = extract_pdf_native_text(file_path)

        if native_ok:
            print("PDF: native text extraction succeeded")
            all_text = native_text
            avg_conf = 95   # native text = near-perfect quality

        else:
            # Strategy 2: render pages → Tesseract
            print("PDF: no native text — trying image OCR via pdf2image + Tesseract")
            images, pdf_error = pdf_to_images(file_path) if TESSERACT_AVAILABLE else ([], tesseract_error)

            if images:
                results = [ocr_image(img) for img in images]
                all_text = "\n\n".join(
                    (f"--- Page {i+1} ---\n" if len(images) > 1 else "") + t.strip()
                    for i, (t, _) in enumerate(results)
                    if t.strip()
                )
                avg_conf = sum(c for _, c in results) / len(results)
                print(f"PDF: Tesseract OCR done, avg_conf={avg_conf:.1f}")
            else:
                print(f"PDF: pdf2image failed ({pdf_error}) — trying Claude Vision")

            # Strategy 3: Claude Vision (BUG FIX — was blocked for PDFs before)
            # Triggered when: pdf2image failed OR Tesseract got low confidence
            if not all_text.strip() or avg_conf < TESSERACT_CONFIDENCE_THRESHOLD:
                print("PDF: using Claude Vision page-by-page fallback")
                claude_text, claude_conf = pdf_pages_via_claude(file_path)
                if claude_text.strip():
                    all_text = claude_text
                    avg_conf = claude_conf
                    used_claude = True
                elif pdf_error:
                    # All 3 strategies failed — return a clear error
                    return {
                        "raw_text": "", "pan_number": None, "name": None, "dob": None,
                        "confidence": 0,
                        "error": f"Could not extract PDF text. {pdf_error}"
                    }

    # ================================================================
    # IMAGE PATH
    # ================================================================
    else:
        image = cv2.imread(file_path)
        if image is None:
            return {
                "raw_text": "", "pan_number": None, "name": None, "dob": None,
                "confidence": 0, "error": f"Could not read image file: {file_path}"
            }
        if TESSERACT_AVAILABLE:
            all_text, avg_conf = ocr_image(image)

        # Claude Vision fallback for low-confidence image OCR
        if avg_conf < TESSERACT_CONFIDENCE_THRESHOLD:
            print(f"Image: Tesseract conf={avg_conf:.1f} — trying Claude Vision")
            claude_text, claude_conf = ocr_via_claude(file_path)
            if claude_text:
                all_text = claude_text
                avg_conf = claude_conf
                used_claude = True
            elif not TESSERACT_AVAILABLE:
                return {
                    "raw_text": "", "pan_number": None, "name": None, "dob": None,
                    "confidence": 0, "error": tesseract_error
                }

    # ================================================================
    # CLEAN + STRUCTURED EXTRACTION
    # ================================================================
    cleaned_text = clean_text(all_text)
    fields = extract_all_fields(cleaned_text)
    confidence = int(min(avg_conf, 100))

    print(f"OCR DONE — engine={'claude-vision' if used_claude else 'tesseract'}, "
          f"conf={confidence}, type={fields.get('document_type')}\n{cleaned_text[:300]}")

    return {
        "raw_text": cleaned_text,
        # top-level shortcuts for backward compatibility
        "pan_number": fields.get("pan_number"),
        "name":       fields.get("name"),
        "dob":        fields.get("dob"),
        # all detected fields
        "fields":     fields,
        "confidence": confidence,
        "ocr_engine": "claude-vision" if used_claude else "tesseract",
    }
