from pathlib import Path

import numpy as np

from input.config import ensemble_tag, moving_frame_d_tag
from data.io import (
    SCATTER_STAT_NAMES,
    load_mf_scatter_all,
    load_mf_scatter_ref,
    mf_scatter_array_to_results,
    mf_scatter_results_to_array,
    save_mf_scatter_all,
    save_mf_scatter_ref,
)


def test_ensemble_tag_format():
    assert ensemble_tag((16, 128, 420, 70)) == "L16M420_EV70"


def test_moving_frame_d_tag_format():
    assert moving_frame_d_tag((0, 0, 1)) == "d001"
    assert moving_frame_d_tag((1, 0, 2)) == "d102"


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
    results = [{"Ks": np.array([1.0, 2.0]), "s": np.array([3.0, 4.0]),
                "k_sq": np.array([5.0, 6.0]), "kcot": np.array([7.0, 8.0])}]
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
