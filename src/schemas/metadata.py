from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocumentType = Literal[
    "research_paper", "book", "report", "documentation", "article", "other"
]


class DocumentMetadata(BaseModel):
    """Metadata describing an *indexed document* (extracted from its text)."""

    title: str = Field(description="The title of the document.")
    author: str | None = Field(
        default=None, description="Main author, exactly as written in the document, or null."
    )
    year: int | None = Field(
        default=None, ge=1000, le=2100, description="Publication year as an integer, or null."
    )
    document_type: DocumentType = Field(
        description="Type of document: research_paper, book, report, documentation, article or other"
    )
    language: str = Field(
        description="Main language of the document as a lowercase ISO 639-1 code, e.g. en, fa"
    )

    @field_validator("language")
    @classmethod
    def _norm_language(cls, value: str) -> str:
        return value.strip().lower()
