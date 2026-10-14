from __future__ import annotations

import dataclasses
import hashlib
import io
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from src.api.deps import build_container
from src.api.main import create_app
from src.config import load_settings
from src.schemas.metadata import DocumentMetadata
from src.schemas.query_filter import QueryMetadataFilter
from src.services.runtime import Runtime

KEY = "sk-or-test-key-1234567890"


class FakeEmbeddings(Embeddings):
    """Deterministic, L2-normalised hash embeddings - no model download needed."""

    dim = 32

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeChat:
    """Stands in for the OpenRouter chat model."""

    def __init__(self) -> None:
        self.query_filter: QueryMetadataFilter | Exception | None = QueryMetadataFilter()
        self.doc_metadata: DocumentMetadata | Exception = DocumentMetadata(
            title="Test Doc", author="Alice", year=2024, document_type="report", language="en"
        )
        self.answer: str = "The answer is 42 [1]."
        self.raise_on_invoke: Exception | None = None
        self.prompts: list = []

    def with_structured_output(self, schema):
        outer = self

        class _S:
            def invoke(self, prompt, **kw):
                value = outer.query_filter if schema is QueryMetadataFilter else outer.doc_metadata
                if isinstance(value, Exception):
                    raise value
                return value

        return _S()

    def invoke(self, messages, **kw):
        self.prompts.append(messages)
        if self.raise_on_invoke:
            raise self.raise_on_invoke
        return SimpleNamespace(content=self.answer)


@pytest.fixture
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def container(tmp_path, fake_chat):
    settings = dataclasses.replace(
        load_settings(),
        vectordb_path=tmp_path / "db",
        collection_name="TestDB",
        openrouter_api_key=KEY,
        upload_max_mb=1,
        embedding_model="fake-model",
    )
    runtime = Runtime(settings, embeddings_loader=lambda: FakeEmbeddings(), chat_model_factory=lambda m, k: fake_chat)
    return build_container(settings, runtime)


@pytest.fixture
def client(container):
    with TestClient(create_app(container), raise_server_exceptions=False) as c:
        yield c


SAMPLE_TEXT = (
    "Retrieval augmented generation combines search with language models. "
    "It grounds answers in retrieved passages. "
    "Metadata filters narrow the search to the right documents. "
    "This improves precision for large libraries."
)


def txt(name: str = "a.txt", text: str = SAMPLE_TEXT):
    return ("files", (name, io.BytesIO(text.encode()), "text/plain"))


def wait_job(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "completed_with_errors", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")
