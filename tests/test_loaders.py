import io

from src.loaders import DocumentLoaderPipeline
from src.loaders.pipeline.base import BasePipelineLoader


def test_extension_registry_matches_declared_support():
    assert set(DocumentLoaderPipeline()._loaders) == set(BasePipelineLoader.supported_extensions())
    assert ".docx" in DocumentLoaderPipeline()._loaders  # was the typo '.docs'


def test_docx_is_routed_to_loader(tmp_path):
    import docx

    d = docx.Document()
    d.add_paragraph("Hello from a Word file.")
    p = tmp_path / "w.docx"
    d.save(p)
    docs = DocumentLoaderPipeline().load(p)
    assert "Hello from a Word file" in docs[0].page_content


def test_txt_and_csv_loaders(tmp_path):
    (tmp_path / "a.txt").write_text("plain text content", encoding="utf-8")
    (tmp_path / "b.csv").write_text("name,age\nann,3\n", encoding="utf-8")
    pipe = DocumentLoaderPipeline()
    assert "plain text" in pipe.load(tmp_path / "a.txt")[0].page_content
    assert "ann" in pipe.load(tmp_path / "b.csv")[0].page_content
