"""Shared fixtures for config and channel resolution tests."""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from input.config import (
    BuildConfig,
    ChanMom,
    Config,
    ScatteringChanMatch,
    meson_chan_index,
    ratio_point_by_label,
    resolve_scattering_chan,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def tcccc_builder() -> BuildConfig:
    return BuildConfig("Tcccc6600")


@pytest.fixture
def tcccc_tetra(tcccc_builder: BuildConfig) -> Config:
    return tcccc_builder.build_config_from_control("tetraquark")


@pytest.fixture
def tcccc_meson(tcccc_builder: BuildConfig) -> Config:
    return tcccc_builder.build_config_from_control("meson")


@pytest.fixture
def x3872_tetra() -> Config:
    return BuildConfig("X3872").build_config_from_control("tetraquark")


@pytest.fixture
def zc3900_tetra() -> Config:
    return BuildConfig("Zc3900").build_config_from_control("tetraquark")


def assert_scattering_labels(
    config: Config,
    match: ScatteringChanMatch,
    *,
    meson_a: str,
    meson_b: str,
    tetra: str,
) -> None:
    """Assert channel indices match LaTeX labels from ENSEMBLE_DB."""
    meson_names = config.meson_chan_name_list
    tetra_names = config.tetra_chan_name_list
    assert meson_names[match.meson_a] == meson_a
    assert meson_names[match.meson_b] == meson_b
    assert tetra_names[match.tetra] == tetra


def assert_ratio_point_labels(
    config: Config,
    point: ChanMom,
    *,
    tetra: str,
    meson_a: str,
    meson_b: str | None = None,
) -> None:
    """Assert ratio lookup key resolves to expected meson labels."""
    from analysis.fit_tmin import ratio_scan_lookup

    target = ratio_scan_lookup(config)[point.key]
    meson_names = config.meson_chan_name_list
    tetra_names = config.tetra_chan_name_list
    assert tetra_names[point.chan] == tetra
    assert meson_names[target.meson_chan] == meson_a
    if meson_b is not None:
        assert meson_names[target.meson_b_chan] == meson_b


def meson_idx(config: Config, label: str) -> int:
    return meson_chan_index(label, config.meson_chan_name_list)


def tetra_key(config: Config, label: str, mom: int) -> tuple[int, int]:
    return ratio_point_by_label(config, label, mom).key
