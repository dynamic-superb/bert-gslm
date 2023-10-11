import abc


class ExtractorBase(abc.ABC):
    """Abstract base extractor class."""

    @abc.abstractmethod
    def extract_feat(self) -> None:
        raise NotImplementedError
