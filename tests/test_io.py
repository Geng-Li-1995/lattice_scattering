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


def test_resample_path_tags():
    config = BuildConfig("Tcccc6600").build_config_from_control("meson")
    path = resample_en_path(config, "meson", config.ensemble_key)
    assert path.name == f"resample_En_meson_{config.tag_name}.npy"

    x3872 = BuildConfig("X3872").build_config_from_control("tetraquark")
    x3872 = replace(x3872, MF_zeta_lamda=50)
    mf = mf_scatter_path(x3872, (16, 128, 420, 70))
    assert mf.name == "resample_scatter_MF_d001_lam50_L16M420_EV70.npy"


def test_mf_scatter_roundtrip(tmp_path: Path):
    results = [
        {name: np.linspace(i, i + 0.1, 5) for name in SCATTER_STAT_NAMES}
        for i in range(2)
    ]
    array = mf_scatter_results_to_array(results)
    restored = mf_scatter_array_to_results(array)
    for level in range(2):
        for name in SCATTER_STAT_NAMES:
            assert np.array_equal(restored[level][name], results[level][name])

    file_path = tmp_path / "scatter.npy"
    payload = [{"Ks": np.array([1.0]), "s": np.array([2.0]), "k_sq": np.array([3.0]), "kcot": np.array([4.0])}]
    save_mf_scatter_all(file_path, payload)
    assert np.array_equal(load_mf_scatter_all(file_path)[0]["kcot"], payload[0]["kcot"])

    ref_path = tmp_path / "ref.npy"
    k_sq, kcot = np.linspace(0, 1, 10), np.linspace(-1, 1, 10)
    save_mf_scatter_ref(ref_path, k_sq, kcot)
    loaded = load_mf_scatter_ref(ref_path)
    assert np.array_equal(loaded[0], k_sq) and np.array_equal(loaded[1], kcot)
