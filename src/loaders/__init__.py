from .docs import DocsLoader
from .web import WebLoader
from .csv import CsvLoader
from .text import TxtLoader
from .markdown import MarkdownLoader
from .pdf import PDFLoader
from .ocr import OcrLoader
from .pipeline.loader_pipeline import DocumentLoaderPipeline

__all__ = [
    "DocsLoader",
    "WebLoader",
    "CsvLoader",
    "TxtLoader",
    "MarkdownLoader",
    "PDFLoader",
    "OcrLoader",
    "DocumentLoaderPipeline",
]
