import pathlib
from abc import ABC, abstractmethod
from typing import ClassVar

from langchain_core.documents import Document


class BasePipelineLoader(ABC):
    """Abstract base class for pipeline document loaders.

    Defines a common interface and shared utilities for loading documents
    from files or directories into LangChain ``Document`` objects.

    Subclasses are required to implement :meth:`load` to define how
    documents are actually read and parsed. This base class additionally
    provides helpers to inspect paths (files, directories, zip archives)
    and to check whether a given path is supported based on its file
    extension.

    Attributes
    ----------
    _SUPPORTED_EXTENSIONS : ClassVar[tuple of str]
        Immutable tuple of file extensions (including the leading dot)
        supported by this loader, e.g. ``(".md", ".pdf")``. Subclasses
        may override this to declare their own supported formats.
    """

    _SUPPORTED_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".md", ".markdown", ".csv", ".pdf", ".docx", ".txt",
    )

    @abstractmethod
    def load(self, path: pathlib.Path | str) -> list[Document]:
        """Load documents from the given path.

        Parameters
        ----------
        path : pathlib.Path or str
            The file or directory path from which documents should be loaded.

        Returns
        -------
        list of Document
            A list of LangChain ``Document`` objects parsed from the given path.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    def is_directory(self, path: pathlib.Path | str) -> bool:
        """Check whether the given path points to a directory.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to check.

        Returns
        -------
        bool
            ``True`` if ``path`` is an existing directory, ``False`` otherwise.
        """
        return pathlib.Path(path).is_dir()

    def is_file(self, path: pathlib.Path | str) -> bool:
        """Check whether the given path points to a regular file.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to check.

        Returns
        -------
        bool
            ``True`` if ``path`` is an existing regular file, ``False`` otherwise.
        """
        return pathlib.Path(path).is_file()

    @classmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        """Get the tuple of file extensions supported by this loader.

        Returns
        -------
        tuple of str
            The extensions supported by this loader, including the leading
            dot (e.g. ``(".md", ".pdf")``). Taken from the class-level
            :attr:`_SUPPORTED_EXTENSIONS` attribute.
        """
        return cls._SUPPORTED_EXTENSIONS

    def can_handle(self, path: pathlib.Path | str) -> bool:
        """Determine whether this loader can handle the given path.

        The decision is based on the file extension of ``path`` compared,
        case-insensitively, against :attr:`supported_extensions`.

        Parameters
        ----------
        path : pathlib.Path or str
            The path whose extension should be validated.

        Returns
        -------
        bool
            ``True`` if the path's extension is supported, ``False`` otherwise.
        """
        return pathlib.Path(path).suffix.lower() in self.supported_extensions()

    def _is_zip(self, path: pathlib.Path | str) -> bool:
        """Check whether the given path has a ``.zip`` extension.

        The comparison is case-insensitive, so ``"archive.ZIP"`` is also
        recognized as a zip file.

        Parameters
        ----------
        path : pathlib.Path or str
            The path to check.

        Returns
        -------
        bool
            ``True`` if the path ends with ``.zip`` (case-insensitive),
            ``False`` otherwise.
        """
        return pathlib.Path(path).suffix.lower() == ".zip"

    def _load_directory(self, path: pathlib.Path | str) -> list[Document]:
        pass


    def load_zip(self, path: pathlib.Path | str) -> list[Document]:
        pass

    def _load_file(self, path: pathlib.Path | str) -> list[Document]:
        pass
