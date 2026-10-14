"""Upload validation and safe staging. User file names never become paths."""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.errors import FileValidationError
from src.loaders.ocr.ocr_loader import IMAGE_EXTENSIONS
from src.loaders.pipeline.base import BasePipelineLoader

DIGITAL_EXTENSIONS = frozenset(BasePipelineLoader.supported_extensions())
SCANNED_EXTENSIONS = frozenset({".pdf"}) | IMAGE_EXTENSIONS

_MAX_NAME = 150
_BAD = re.compile(r"[^\w.\- ()\u0600-\u06FF]+", re.UNICODE)
_CHUNK = 1024 * 1024


@dataclass
class StagedFile:
    path: Path
    original_name: str
    size: int
    sha256: str

    @property
    def doc_id(self) -> str:
        return self.sha256[:32]


def sanitize_filename(name: str | None) -> str:
    base = (name or "").replace("\\", "/").split("/")[-1]
    base = _BAD.sub("_", base).strip(" .")
    if len(base) > _MAX_NAME:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: _MAX_NAME - len(ext) - 1] + dot + ext) if dot else base[:_MAX_NAME]
    return base or "unnamed"


def _sniff(head: bytes, ext: str) -> bool:
    if ext == ".pdf":
        return head.lstrip()[:5] == b"%PDF-"
    if ext == ".docx":
        return head[:4] == b"PK\x03\x04"
    if ext in (".png",):
        return head[:8] == b"\x89PNG\r\n\x1a\n"
    if ext in (".jpg", ".jpeg"):
        return head[:3] == b"\xff\xd8\xff"
    if ext in (".tif", ".tiff"):
        return head[:4] in (b"II*\x00", b"MM\x00*")
    if ext == ".bmp":
        return head[:2] == b"BM"
    if ext == ".webp":
        return head[:4] == b"RIFF" and head[8:12] == b"WEBP"
    if ext in (".txt", ".md", ".markdown", ".csv"):
        return b"\x00" not in head[:4096]
    return False


def stage_upload(
    fileobj: BinaryIO,
    filename: str | None,
    dest_dir: Path,
    allowed: frozenset[str],
    max_bytes: int,
) -> StagedFile:
    """Validate and copy an upload into ``dest_dir`` under a random name."""
    clean_name = sanitize_filename(filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in allowed:
        raise FileValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(allowed))}.",
            error_code="UNSUPPORTED_FILE_TYPE",
            status_code=415,
        )
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"
    digest = hashlib.sha256()
    size = 0
    head = b""
    try:
        with open(dest, "wb") as out:
            while True:
                block = fileobj.read(_CHUNK)
                if not block:
                    break
                if not head:
                    head = block[:16]
                size += len(block)
                if size > max_bytes:
                    raise FileValidationError(
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                        error_code="FILE_TOO_LARGE",
                        status_code=413,
                    )
                digest.update(block)
                out.write(block)
        if size == 0:
            raise FileValidationError("The file is empty.", error_code="EMPTY_FILE")
        if not _sniff(head, ext):
            raise FileValidationError(
                f"The file content does not look like a valid '{ext}' file.",
                error_code="FILE_CONTENT_MISMATCH",
                status_code=415,
            )
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return StagedFile(path=dest, original_name=clean_name, size=size, sha256=digest.hexdigest())
