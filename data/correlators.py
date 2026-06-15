"""Lightweight wrappers for meson / tetraquark correlator arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_AXIS = -1


def _as_f64(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


@dataclass
class Correlator4D:
    """[channel, momentum, time, sample]; momentum index = n^2 on the lattice."""

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = _as_f64(self.data)
        if self.data.ndim != 4:
            raise ValueError(f"Correlator4D expects 4 axes, got shape {self.data.shape}")

    def __getitem__(self, ch: int) -> np.ndarray:
        return self.data[ch]

    def at(self, ch: int, mom: int) -> np.ndarray:
        return self.data[ch, mom]


@dataclass
class TetraquarkCorrelator:
    """[ch_src, mom_src, ch_snk, mom_snk, time, sample]; mom indices = n^2."""

    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = _as_f64(self.data)
        if self.data.ndim != 6:
            raise ValueError(f"TetraquarkCorrelator expects 6 axes, got shape {self.data.shape}")

    def at(self, ch_src: int, mom_src: int, ch_snk: int, mom_snk: int) -> np.ndarray:
        return self.data[ch_src, mom_src, ch_snk, mom_snk]

    @property
    def n_time(self) -> int:
        return self.data.shape[-2]

    @property
    def n_sample(self) -> int:
        return self.data.shape[-1]


@dataclass
class RawCorrelators:
    meson: Correlator4D | None = None
    tetraquark: TetraquarkCorrelator | None = None

    @classmethod
    def from_parts(
        cls,
        *,
        meson: np.ndarray | Correlator4D | None = None,
        tetraquark: np.ndarray | TetraquarkCorrelator | None = None,
    ) -> RawCorrelators:
        return cls(
            meson=meson if isinstance(meson, Correlator4D) else Correlator4D(meson) if meson is not None else None,
            tetraquark=(
                tetraquark
                if isinstance(tetraquark, TetraquarkCorrelator)
                else TetraquarkCorrelator(tetraquark) if tetraquark is not None else None
            ),
        )

    def array_items(self) -> dict[str, np.ndarray]:
        items: dict[str, np.ndarray] = {}
        if self.meson is not None:
            items["meson"] = self.meson.data
        if self.tetraquark is not None:
            items["tetraquark"] = self.tetraquark.data
        return items


@dataclass
class AnalysisCorrelators:
    """Correlators ready for fitting (tetraquark is 4D after GEVP)."""

    meson: Correlator4D | None = None
    tetraquark: Correlator4D | None = None

    @classmethod
    def from_raw(cls, raw: RawCorrelators) -> AnalysisCorrelators:
        return cls(meson=raw.meson, tetraquark=None)

    def with_tetraquark(self, data: np.ndarray | Correlator4D) -> AnalysisCorrelators:
        tetra = data if isinstance(data, Correlator4D) else Correlator4D(data)
        return AnalysisCorrelators(meson=self.meson, tetraquark=tetra)

    def active(self, is_meson_analysis: bool, is_tetraquark_analysis: bool) -> Correlator4D:
        if is_meson_analysis:
            if self.meson is None:
                raise ValueError("Meson correlator not loaded.")
            return self.meson
        if is_tetraquark_analysis:
            if self.tetraquark is None:
                raise ValueError("Tetraquark correlator not loaded.")
            return self.tetraquark
        raise ValueError("No analysis mode selected.")
