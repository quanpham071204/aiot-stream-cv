"""Pluggable frame analyzers (foam coverage, overflow, room for later features).

Each analyzer sees every processed frame and returns an ``AnalyzerResult``:
- ``measurement``: a continuous value shown on the dashboard (e.g. foam %), or None;
- ``events``: alert payloads stored as events (e.g. one ``overflow`` per spill).

Analyzers are switched on/off per source in config::

    analyzers:
      foam_coverage: {enabled: true, roi: [[x, y], [x, y], [x, y], [x, y]]}
      overflow:      {enabled: true, roi: [[x, y], ...], cooldown_s: 30}

Adding a feature = subclass ``Analyzer`` and add it to ``registry()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnalyzerResult:
    measurement: Optional[dict] = None
    events: list[dict] = field(default_factory=list)


class Analyzer:
    """Base class. Subclasses set ``name`` (and ``event_type`` if they raise alerts)."""

    name = "analyzer"
    event_type: Optional[str] = None

    def process(self, frame: np.ndarray, t: float) -> AnalyzerResult:  # pragma: no cover - interface
        raise NotImplementedError

    def draw(self, frame: np.ndarray) -> None:
        """Draw ROI / state onto the annotated frame in place (optional)."""


def registry() -> dict[str, type[Analyzer]]:
    from .foam import FoamCoverage
    from .overflow import Overflow

    return {cls.name: cls for cls in (FoamCoverage, Overflow)}


def build_analyzers(cfg: Optional[dict]) -> list[Analyzer]:
    """Create the analyzers enabled in a source's ``analyzers`` config block."""
    known = registry()
    out: list[Analyzer] = []
    for name, opts in (cfg or {}).items():
        opts = dict(opts or {})
        if not opts.pop("enabled", True):
            continue
        if name not in known:
            raise ValueError(f"unknown analyzer {name!r}; known: {sorted(known)}")
        out.append(known[name](**opts))
    return out
