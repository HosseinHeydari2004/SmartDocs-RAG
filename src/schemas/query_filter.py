"""Schema for metadata constraints found in a *user query* (not a document)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .metadata import DocumentType

_YEAR_MIN, _YEAR_MAX = 1000, 2100


class QueryMetadataFilter(BaseModel):
    """Constraints the user *explicitly* stated. Unstated fields must stay null."""

    filename: str | None = Field(
        default=None, description="Exact file name the user wants to search in, e.g. report.pdf."
    )
    author: str | None = Field(default=None, description="Author name explicitly mentioned.")
    document_type: DocumentType | None = Field(
        default=None, description="Document type, only if explicitly mentioned."
    )
    language: str | None = Field(
        default=None, description="Lowercase ISO 639-1 language code, only if explicitly mentioned."
    )
    year: int | None = Field(default=None, description="Exactly one publication year.")
    year_gte: int | None = Field(default=None, description="Published in or after this year.")
    year_lte: int | None = Field(default=None, description="Published in or before this year.")
    years: list[int] | None = Field(
        default=None, description="Several specific publication years (at most 10)."
    )

    @field_validator("filename", "author", "language", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("language")
    @classmethod
    def _lower(cls, value: str | None) -> str | None:
        return value.lower() if value else value

    @field_validator("year", "year_gte", "year_lte")
    @classmethod
    def _year_range(cls, value: int | None) -> int | None:
        if value is not None and not (_YEAR_MIN <= value <= _YEAR_MAX):
            raise ValueError(f"year must be between {_YEAR_MIN} and {_YEAR_MAX}")
        return value

    @field_validator("years")
    @classmethod
    def _years(cls, value: list[int] | None) -> list[int] | None:
        if not value:
            return None
        if len(value) > 10:
            raise ValueError("at most 10 years are allowed")
        for y in value:
            if not (_YEAR_MIN <= y <= _YEAR_MAX):
                raise ValueError(f"year must be between {_YEAR_MIN} and {_YEAR_MAX}")
        return sorted(set(value))

    @model_validator(mode="after")
    def _consistent(self) -> "QueryMetadataFilter":
        if self.year_gte is not None and self.year_lte is not None and self.year_gte > self.year_lte:
            raise ValueError("year_gte must not be greater than year_lte")
        return self

    def active(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def is_empty(self) -> bool:
        return not self.active()

    def merged_with(self, manual: "QueryMetadataFilter | None") -> "QueryMetadataFilter":
        """Combine LLM-extracted filter with the user's manual filter.

        Fields set manually always win; fields only the LLM found are kept, so a
        manual choice never silently drops a restriction and vice versa.
        """
        if manual is None:
            return self.model_copy()
        data = self.active()
        data.update(manual.active())
        return QueryMetadataFilter(**data)
