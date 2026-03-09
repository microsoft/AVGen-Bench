from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationArtifact:
    data: bytes
    extension: str


class BaseGenerationClient(ABC):
    """Base interface for prompt-based generation clients."""

    @abstractmethod
    def video_generation(self, prompt: str, **kwargs) -> GenerationArtifact:
        """Generate a video artifact from one prompt."""

    def image_generation(self, prompt: str, **kwargs) -> GenerationArtifact:
        """
        Generate an image artifact from one prompt.

        Keep this interface for future model integrations that need image generation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image_generation()."
        )

