"""Abstract base class for 3D generation engines."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from pathlib import Path


class BaseEngine(ABC):
    """Base interface that all 3D generation engines must implement.

    This allows the system to support multiple model backends
    (Trellis, Hunyuan, TripoSG, etc.) through a uniform API.

    Engine-specific parameters are passed via **engine_params.
    Each engine extracts the params it needs and ignores the rest.
    """

    name: str = "base"
    loaded: bool = False

    @abstractmethod
    def load(self, weights_dir: str, device: str = "cuda") -> None:
        """Load model weights into GPU memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model weights from GPU memory."""
        ...

    @abstractmethod
    def generate_from_image(
        self,
        image_path: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        """Generate 3D from a single image.

        Returns dict with engine-specific generation data.
        """
        ...

    @abstractmethod
    def generate_from_images(
        self,
        image_paths: list[str],
        seed: int,
        mode: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        """Generate 3D from multiple images of the same object."""
        ...

    @abstractmethod
    def generate_from_text(
        self,
        prompt: str,
        seed: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict:
        """Generate 3D from a text prompt."""
        ...

    @abstractmethod
    def export_mesh(
        self,
        generation_data: dict,
        output_dir: str,
        formats: list[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        **engine_params,
    ) -> dict[str, Path]:
        """Export generation result to mesh files.

        Returns dict mapping format name -> output file path.
        """
        ...

    @abstractmethod
    def render_preview(
        self,
        generation_data: dict,
        output_path: str,
        resolution: int = 512,
        num_frames: int = 120,
    ) -> str:
        """Render a preview video of the generation. Returns path to video."""
        ...
