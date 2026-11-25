"""Backend registry: ``fake`` (synthetic WAVs) is the remaining generator."""
from __future__ import annotations

from mixgen.backends.base import GenBackend, GenBackendError
from mixgen.plan import Plan

BACKENDS = ("fake",)


def make_backend(plan: Plan, name: str, **kwargs) -> GenBackend:
    if name == "fake":
        from mixgen.backends.fake import FakeBackend

        return FakeBackend(sample_rate=plan.sample_rate)
    raise GenBackendError(f"unknown backend {name!r}; choose one of {BACKENDS}")


__all__ = ["BACKENDS", "GenBackend", "GenBackendError", "make_backend"]
