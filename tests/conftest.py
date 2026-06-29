"""Shared fixtures for config and channel resolution tests."""

from __future__ import annotations

import pytest

from input.config import (
    BuildConfig,
    Config,
    ScatteringChanMatch,
    meson_chan_index,
    ratio_point_by_label,
)


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(0)


@pytest.fixture
def tcccc_tetra() -> Config:
    return BuildConfig("Tcccc6600").build_config_from_control("tetraquark")


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
    assert config.meson_chan_name_list[match.meson_a] == meson_a
    assert config.meson_chan_name_list[match.meson_b] == meson_b
    assert config.tetra_chan_name_list[match.tetra] == tetra


def meson_idx(config: Config, label: str) -> int:
    return meson_chan_index(label, config.meson_chan_name_list)


def tetra_key(config: Config, label: str, mom: int) -> tuple[int, int]:
    return ratio_point_by_label(config, label, mom).key
