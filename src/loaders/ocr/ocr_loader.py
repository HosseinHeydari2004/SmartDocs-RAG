from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from pytesseract import image_to_string
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from src.ocr_env import configure_tesseract, find_poppler_path

from ..base import BaseLoader

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


class OcrError(RuntimeError):
    """Raised when OCR cannot be performed at all (engine missing, unreadable input)."""


@dataclass
class OcrReport:
    """What happened during the last ``OcrLoader.load`` call."""

    pages_total: int = 0
    pages_with_text: int = 0
    pages_empty: list[int] = field(default_factory=list)
    pages_failed: list[int] = field(default_factory=list)


class OcrLoader(BaseLoader):
    """OCR loader for scanned PDFs and images (Tesseract).

    One ``Document`` is returned per page that produced text. Pages are rasterised
    and recognised one at a time, so memory stays bounded for long PDFs.
    Tables, formulas and charts are not interpreted - only raw text is extracted.
    """

    def __init__(self) -> None:
        self.last_report = OcrReport()

    def load(
        self,
        path: Path | str,
        lang: str = "eng",
        dpi: int = 300,
        max_pages: int | None = None,
        on_page: Callable[[int, int], None] | None = None,
        timeout: int = 120,
    ) -> list[Document]:
        path = Path(path)
        self.last_report = OcrReport()
        configure_tesseract()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path, lang, dpi, max_pages, on_page, timeout)
        if suffix in IMAGE_EXTENSIONS:
            return self._load_image(path, lang, on_page, timeout)
        raise OcrError(f"Unsupported file type for OCR: {suffix or '(none)'}")

    # ------------------------------------------------------------------ helpers
    def _document(self, path: Path, page_number: int, text: str) -> Document:
        return Document(
            page_content=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "page_number": page_number,
                "extraction": "ocr",
            },
        )

    def _recognise(self, image: Image.Image, lang: str, timeout: int) -> str:
        try:
            return image_to_string(image, lang=lang, timeout=timeout).strip()
        except TesseractNotFoundError as exc:
            raise OcrError("Tesseract is not installed or not on PATH.") from exc

    def _load_pdf(self, path, lang, dpi, max_pages, on_page, timeout) -> list[Document]:
        try:
            total = int(pdfinfo_from_path(str(path), poppler_path=find_poppler_path())["Pages"])
        except Exception as exc:  # poppler missing, corrupt or encrypted PDF
            raise OcrError("Could not read the PDF (is Poppler installed? is the file valid?).") from exc
        if max_pages is not None and total > max_pages:
            raise OcrError(f"PDF has {total} pages; the OCR limit is {max_pages}.")

        report = self.last_report
        report.pages_total = total
        docs: list[Document] = []
        for page_number in range(1, total + 1):
            try:
                images = convert_from_path(
                    str(path), dpi=dpi, first_page=page_number, last_page=page_number,
                    poppler_path=find_poppler_path(),
                )
                text = self._recognise(images[0], lang, timeout)
            except OcrError:
                raise
            except (TesseractError, RuntimeError, IndexError, OSError):
                report.pages_failed.append(page_number)
                text = ""
            else:
                if text:
                    docs.append(self._document(path, page_number, text))
                    report.pages_with_text += 1
                else:
                    report.pages_empty.append(page_number)
            if on_page:
                on_page(page_number, total)
        if len(report.pages_failed) == total:
            raise OcrError("OCR failed on every page.")
        return docs

    def _load_image(self, path, lang, on_page, timeout) -> list[Document]:
        report = self.last_report
        report.pages_total = 1
        try:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            raise OcrError("The image could not be read.") from exc
        try:
            text = self._recognise(rgb, lang, timeout)
        except TesseractError as exc:
            report.pages_failed.append(1)
            raise OcrError("OCR failed on the image.") from exc
        docs: list[Document] = []
        if text:
            docs.append(self._document(path, 1, text))
            report.pages_with_text = 1
        else:
            report.pages_empty.append(1)
        if on_page:
            on_page(1, 1)
        return docs
