from abc import ABC, abstractmethod
from pathlib import Path
from langchain_core.documents import Document


class BaseLoader(ABC):
    """base loader for all loaders"""

    @abstractmethod
    def load(self, path: Path | str) -> list[Document]:
        pass
