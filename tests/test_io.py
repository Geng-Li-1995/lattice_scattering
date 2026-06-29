from dataclasses import replace
from pathlib import Path

import numpy as np

from data.io import (
    SCATTER_STAT_NAMES,
    load_mf_scatter_all,
    load_mf_scatter_ref,
    mf_scatter_array_to_results,
    mf_scatter_path,
    mf_scatter_results_to_array,
    resample_en_path,
    save_mf_scatter_all,
    save_mf_scatter_ref,
)
from input.config import BuildConfig


def test_mf_scatter_path_naming():
    config = BuildConfig("X3872").build_config_from_control("tetraquark")
    config = replace(config, MF_zeta_lamda=50)
    path = mf_scatter_path(config, (16, 128, 420, 70))
    assert path.name == "resample_scatter_MF_d001_lam50_L16M420_EV70.npy"
    assert path.parent.name == "resampled"


def test_resample_en_path_matches_read_write():
    config = BuildConfig("Tcccc6600").build_config_from_control("meson")
    path = resample_en_path(config, "meson", config.ensemble_key)
    assert path.name == f"resample_En_meson_{config.tag_name}.npy"
    assert path.parent.name == "resampled"


def test_mf_scatter_cache_roundtrip():
    n_level, n_sample = 2, 5
    results = [
        {name: np.linspace(i, i + 0.1, n_sample) for name in SCATTER_STAT_NAMES}
        for i in range(n_level)
    ]
    array = mf_scatter_results_to_array(results)
    assert array.shape == (n_level, n_sample, len(SCATTER_STAT_NAMES))

    restored = mf_scatter_array_to_results(array)
    for level in range(n_level):
        for name in SCATTER_STAT_NAMES:
            assert np.array_equal(restored[level][name], results[level][name])


def test_mf_scatter_file_roundtrip(tmp_path: Path):
    path = tmp_path / "scatter.npy"
    results = [
        {
            "Ks": np.array([1.0, 2.0]),
            "s": np.array([3.0, 4.0]),
            "k_sq": np.array([5.0, 6.0]),
            "kcot": np.array([7.0, 8.0]),
        }
    ]
    save_mf_scatter_all(path, results)
    loaded = load_mf_scatter_all(path)
    assert loaded is not None
    assert np.array_equal(loaded[0]["kcot"], results[0]["kcot"])


def test_mf_scatter_ref_file_roundtrip(tmp_path: Path):
    path = tmp_path / "ref.npy"
    k_sq = np.linspace(0.0, 1.0, 10)
    kcot = np.linspace(-1.0, 1.0, 10)
    save_mf_scatter_ref(path, k_sq, kcot)
    loaded = load_mf_scatter_ref(path)
    assert loaded is not None
    assert np.array_equal(loaded[0], k_sq)
    assert np.array_equal(loaded[1], kcot)
