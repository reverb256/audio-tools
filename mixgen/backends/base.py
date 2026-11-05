"""Generation backend interface.

A backend turns one track into its audio takes. Stages call it with temporary
paths and commit the results atomically, so a crash mid-generation never leaves
a half-written take in place.

Backends that generate tracks as one batch (``acestep``: one gpu-music-job with
``--takes N``) override :meth:`generate_track`; per-take backends only need
:meth:`generate`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from mixgen.plan import Track


class GenBackendError(RuntimeError):
    """Backend cannot run: missing configuration, missing wrapper, failed job, ..."""


class GenBackend(ABC):
    name = "base"

    @abstractmethod
    def generate(self, track: Track, take_idx: int, out_path: Path) -> Path:
        """Produce take ``take_idx`` (1-based) for ``track`` at ``out_path``."""
        raise NotImplementedError

    def generate_track(self, track: Track, targets: list[tuple[int, Path]],
                       *, force: bool = False) -> list[Path]:
        """Produce the requested ``(take_idx, out_path)`` pairs for one track.

        Default is one call to :meth:`generate` per take. Batch backends submit
        a single job for the whole track and copy each requested take out of it.
        """
        return [self.generate(track, take_idx, out_path) for take_idx, out_path in targets]
