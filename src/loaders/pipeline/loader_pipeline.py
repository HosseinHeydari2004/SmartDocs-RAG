from __future__ import annotations

import pathlib
import tempfile
import zipfile

from tqdm import tqdm
from langchain_core.documents import Document

from .base import BasePipelineLoader
from ..base import BaseLoader
from src.loaders import (
    DocsLoader,
    TxtLoader,
    PDFLoader,
    MarkdownLoader,
    CsvLoader,
)


class DocumentLoaderPipeline(BasePipelineLoader):
    """
    A pipeline loader that orchestrates the loading of documents from various
    sources (files, directories, and ZIP archives) by dispatching to the
    appropriate loader based on file extension.

    This pipeline extends :class:`BasePipelineLoader` and provides a unified
    interface for loading documents from a given path. It automatically detects
    whether the given path is a file, a directory, or a ZIP archive, and routes
    the loading process accordingly. Supported file extensions include
    ``.docx``, ``.pdf``, ``.txt``, ``.md``, ``.markdown`` and ``.csv``.

    Attributes
    ----------
    _loaders : dict of str to type[BaseLoader]
        Mapping of supported file extensions (lowercase, including the leading
        dot) to their corresponding loader classes. Initialized with the
        following entries:

        - ``".docx"`` -> ``DocsLoader``
        - ``".pdf"``  -> ``PDFLoader``
        - ``".txt"``  -> ``TxtLoader``
        - ``".md"``   -> ``MarkdownLoader``
        - ``".csv"``  -> ``CsvLoader``

    Examples
    --------
    >>> from pathlib import Path
    >>> from document_loader_pipeline import DocumentLoaderPipeline
    >>>
    >>> pipeline = DocumentLoaderPipeline()
    >>>
    >>> # Load from a single file
    >>> docs = pipeline.load(Path("report.pdf"))
    >>>
    >>> # Load from a directory (recursive)
    >>> docs = pipeline.load(Path("./data"))
    >>>
    >>> # Load from a ZIP archive
    >>> docs = pipeline.load(Path("archive.zip"))

    See Also
    --------
    BasePipelineLoader : The abstract base class for pipeline loaders.
    BaseLoader : The base class for individual file loaders.
    DocsLoader, TxtLoader, PDFLoader, MarkdownLoader, CsvLoader :
        Concrete loader implementations for specific file types.
    langchain_core.documents.Document :
        The document object returned by loaders.
    """

    def __init__(self) -> None:
        """
        Initialize the :class:`DocumentLoaderPipeline` with its default set of
        supported file extensions and their corresponding loader classes.
        """
        self._loaders: dict[str, type[BaseLoader]] = {
            ".docx": DocsLoader,
            ".pdf": PDFLoader,
            ".txt": TxtLoader,
            ".md": MarkdownLoader,
            ".markdown": MarkdownLoader,
            ".csv": CsvLoader,
        }

    def load(
            self,
            path: pathlib.Path | str,
    ) -> list[Document]:
        """
        Load documents from the given path.

        The dispatch logic is as follows:

        1. If ``path`` is a directory, delegates to :meth:`_load_directory`.
        2. If ``path`` is a ZIP archive, delegates to :meth:`_load_zip`.
        3. If ``path`` is a file, delegates to :meth:`_load_file`.
        4. Otherwise, raises :class:`ValueError`.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to a file, directory, or ZIP archive to load documents
            from.

        Returns
        -------
        list of Document
            A list of :class:`langchain_core.documents.Document` objects
            extracted from the path.

        Raises
        ------
        ValueError
            If the path is not a directory, a ZIP archive, or a supported
            file.
        """
        if self.is_directory(path=path):
            return self._load_directory(path=path)

        if self._is_zip(path=path):
            return self._load_zip(path=path)

        if self.is_file(path=path):
            return self._load_file(path=path)

        raise ValueError(f"Unable to load path: {path}")

    def _load_file(
            self,
            path: pathlib.Path | str,
    ) -> list[Document]:
        """
        Load documents from a single file based on its extension.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to a supported file.

        Returns
        -------
        list of Document
            A list of documents extracted from the file.

        Raises
        ------
        ValueError
            If the file extension is not supported (i.e., not present in
            ``_loaders``).
        """
        path = pathlib.Path(path)
        extension = path.suffix.lower()

        loader_cls = self._loaders.get(extension)

        if loader_cls is None:
            raise ValueError(
                f"Unsupported file extension: {extension}"
            )

        return loader_cls().load(path=path)

    def _load_directory(
            self,
            path: pathlib.Path | str,
            *,
            show_progress: bool = True,
    ) -> list[Document]:
        """
        Recursively load documents from all supported files within a
        directory.

        Only files with extensions present in ``_loaders`` or ``.zip`` are
        considered. ZIP files encountered within the directory are recursively
        loaded via :meth:`_load_zip`. :class:`ValueError` exceptions raised
        during individual file loading are silently ignored (the file is
        skipped).

        Parameters
        ----------
        path : pathlib.Path or str
            The directory to scan for supported files.
        show_progress : bool, optional
            Whether to display a ``tqdm`` progress bar during loading.
            Default is ``True``.

        Returns
        -------
        list of Document
            A list of documents extracted from all supported files in the
            directory.
        """
        path = pathlib.Path(path)
        documents: list[Document] = []

        # Only collect supported files.
        files = [
            child
            for child in path.rglob("*")
            if child.is_file()
               and (
                       child.suffix.lower() in self._loaders
                       or child.suffix.lower() == ".zip"
               )
        ]

        progress = tqdm(
            files,
            desc="Loading documents",
            unit="file",
            dynamic_ncols=True,
            disable=not show_progress,
            leave=show_progress,
        )

        for file_path in progress:

            progress.set_postfix_str(
                file_path.name,
                refresh=False,
            )

            try:

                if file_path.suffix.lower() == ".zip":
                    documents.extend(
                        self._load_zip(
                            path=file_path,
                            show_progress=False,
                        )
                    )

                else:
                    documents.extend(
                        self._load_file(path=file_path)
                    )

            except ValueError:
                continue

        return documents

    def _load_zip(
            self,
            path: pathlib.Path | str,
            *,
            show_progress: bool = True,
    ) -> list[Document]:
        """
        Extract and load documents from a ZIP archive.

        The ZIP archive is extracted into a temporary directory, which is
        automatically cleaned up after loading. Extraction progress and
        subsequent directory loading progress are both displayed if
        ``show_progress`` is ``True``. Delegates the actual document loading
        to :meth:`_load_directory` on the temporary extraction directory.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to the ZIP archive.
        show_progress : bool, optional
            Whether to display ``tqdm`` progress bars during extraction and
            loading. Default is ``True``.

        Returns
        -------
        list of Document
            A list of documents extracted from all supported files within the
            ZIP archive.
        """
        path = pathlib.Path(path)
        documents: list[Document] = []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)

            with zipfile.ZipFile(path, "r") as zf:
                members = zf.infolist()

                with tqdm(
                        total=len(members),
                        desc=f"Extracting {path.name}",
                        unit="file",
                        dynamic_ncols=True,
                        disable=not show_progress,
                        leave=show_progress,
                ) as progress:
                    for member in members:
                        zf.extract(member, tmp_path)

                        progress.update(1)
                        progress.set_postfix_str(
                            pathlib.Path(member.filename).name,
                            refresh=False,
                        )

            documents.extend(
                self._load_directory(
                    tmp_path,
                    show_progress=show_progress,
                )
            )

        return documents
