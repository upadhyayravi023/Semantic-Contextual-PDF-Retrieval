from asyncio import log
import io
from tkinter import Image
from colorama import Fore, Style
import fitz
import pdfplumber
import pytesseract


def extract_text_from_pdf(filepath):
    log.info(f"{Fore.CYAN}📄 Starting text extraction from PDF: {filepath}{Style.RESET_ALL}")
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            log.info(f"{Fore.YELLOW}→ PDF has {len(pdf.pages)} pages{Style.RESET_ALL}")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    log.info(f"{Fore.GREEN}✔ Extracted text from page {i+1}{Style.RESET_ALL}")
                    text += page_text + "\n"
                else:
                    log.warning(f"{Fore.MAGENTA}⚠ No text found on page {i+1}{Style.RESET_ALL}")
        
        if not text.strip():
            log.warning(f"{Fore.RED}⚙️ Falling back to OCR (scanned PDF detected)...{Style.RESET_ALL}")
            text = ""
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, lang="eng")
                text += ocr_text + "\n"
                log.info(f"{Fore.GREEN}🧠 OCR extracted text from page {page_num+1}{Style.RESET_ALL}")

        log.info(f"{Fore.CYAN}✅ Text extraction completed. Total characters: {len(text)}{Style.RESET_ALL}")
        return text.strip()

    except Exception as e:
        log.error(f"{Fore.RED}❌ Error reading PDF: {e}{Style.RESET_ALL}")
        return ""

