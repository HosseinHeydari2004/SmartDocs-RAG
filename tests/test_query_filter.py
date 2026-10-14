import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.preprocessing.parsing.metadata_extraction import MetadataExtractor
from src.schemas.query_filter import QueryMetadataFilter
from src.vectordb.chroma import VectorStore
from src.vectordb.filters import build_chroma_where
from tests.conftest import FakeEmbeddings


def test_empty_filter_gives_no_where():
    assert build_chroma_where(QueryMetadataFilter()) is None
    assert build_chroma_where(None) is None


def test_single_condition_has_no_and():
    assert build_chroma_where(QueryMetadataFilter(year=2024)) == {"year": {"$eq": 2024}}


def test_multiple_conditions_use_and():
    where = build_chroma_where(QueryMetadataFilter(year_gte=2020, year_lte=2024, language="en", filename="a.pdf"))
    assert set(where) == {"$and"}
    assert {"year": {"$gte": 2020}} in where["$and"] and {"year": {"$lte": 2024}} in where["$and"]
    assert {"language": {"$eq": "en"}} in where["$and"] and {"filename": {"$eq": "a.pdf"}} in where["$and"]


def test_years_list_uses_in():
    assert build_chroma_where(QueryMetadataFilter(years=[2024, 2022, 2024])) == {"year": {"$in": [2022, 2024]}}


@pytest.mark.parametrize("kw", [{"year": 99}, {"year_gte": 2025, "year_lte": 2020}, {"document_type": "poem"}, {"years": list(range(2000, 2012))}])
def test_invalid_values_rejected(kw):
    with pytest.raises(ValidationError):
        QueryMetadataFilter(**kw)


def test_blank_strings_become_none():
    assert QueryMetadataFilter(author="  ", language=" EN ").active() == {"language": "en"}


def test_manual_filter_wins_per_field_and_keeps_others():
    extracted = QueryMetadataFilter(year_gte=2020, language="en")
    manual = QueryMetadataFilter(language="fa", filename="x.pdf")
    merged = extracted.merged_with(manual)
    assert merged.active() == {"year_gte": 2020, "language": "fa", "filename": "x.pdf"}
    assert extracted.merged_with(None).active() == extracted.active()


def test_extractor_converts_persian_digits_and_returns_validated_model(fake_chat):
    fake_chat.query_filter = QueryMetadataFilter(year=2024)
    seen = {}

    class Spy:
        def with_structured_output(self, schema):
            outer = fake_chat.with_structured_output(schema)

            class W:
                def invoke(self, prompt):
                    seen["prompt"] = prompt
                    return outer.invoke(prompt)

            return W()

    out = MetadataExtractor(lambda: Spy()).extract_query_filter("مقاله‌های سال ۲۰۲۴")
    assert out.year == 2024 and "2024" in seen["prompt"] and "۲۰۲۴" not in seen["prompt"]


def test_extractor_rejects_non_model_output():
    class Bad:
        def with_structured_output(self, schema):
            class W:
                def invoke(self, prompt):
                    return None

            return W()

    with pytest.raises(ValueError):
        MetadataExtractor(lambda: Bad()).extract_query_filter("q")


def _store(container):
    return VectorStore(container.settings, embeddings=FakeEmbeddings())


def test_chroma_applies_real_filters(container):
    store = _store(container)
    mk = lambda text, **m: Document(page_content=text, metadata={"doc_id": m["filename"], **m})
    store.add_document(
        [
            mk("deep learning models for vision", filename="a.pdf", year=2024, language="en", document_type="report"),
            mk("deep learning in medicine", filename="b.pdf", year=2021, language="en", document_type="article"),
            mk("deep learning for text", filename="c.pdf", year=2019, language="fa", document_type="report"),
        ],
        ["1", "2", "3"],
    )
    names = lambda where: sorted(d.metadata["filename"] for d in store.search("deep learning", where, k=10))
    assert names(None) == ["a.pdf", "b.pdf", "c.pdf"]
    assert names(build_chroma_where(QueryMetadataFilter(year=2024))) == ["a.pdf"]
    assert names(build_chroma_where(QueryMetadataFilter(year_gte=2020))) == ["a.pdf", "b.pdf"]
    assert names(build_chroma_where(QueryMetadataFilter(year_gte=2020, language="en", document_type="article"))) == ["b.pdf"]
    assert names(build_chroma_where(QueryMetadataFilter(years=[2019, 2021]))) == ["b.pdf", "c.pdf"]
    assert names(build_chroma_where(QueryMetadataFilter(filename="zzz.pdf"))) == []


def test_embedding_model_mismatch_blocks_writes_and_reads(container):
    import dataclasses

    from src.errors import EmbeddingMismatchError

    _store(container).add_document([Document(page_content="x y z", metadata={"doc_id": "d"})], ["1"])
    other = dataclasses.replace(container.settings, embedding_model="another-model")
    store2 = VectorStore(other, embeddings=FakeEmbeddings())
    assert store2.embedding_mismatch
    with pytest.raises(EmbeddingMismatchError):
        store2.search("x")
    with pytest.raises(EmbeddingMismatchError):
        store2.add_document([Document(page_content="q", metadata={})], ["2"])
    store2.reset()  # re-index path
    assert not store2.embedding_mismatch and store2.count_chunks() == 0
