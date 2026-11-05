"""Generation backend interface.

A backend turns one (track, take index) into one audio file at ``out_path``.
Stages call it with a temporary path and commit the result atomically, so a
crash mid-generation never leaves a half-written take in place.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mixgen.plan import Track


class GenBackendError(RuntimeError):
    """Backend cannot run: missing configuration, missing wrapper, timeout, ..."""


class GenBackend(ABC):
    name = "base"

    @abstractmethod
    def generate(self, track: Track, take_idx: int, out_path: Path) -> Path:
        """Produce take ``take_idx`` (1-based) for ``track`` at ``out_path``."""
        raise NotImplementedError
