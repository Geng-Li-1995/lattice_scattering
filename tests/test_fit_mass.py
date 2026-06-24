from types import SimpleNamespace

import numpy as np

from analysis.fit_mass import (
    disp_fit_lookup,
    en_fit_lookup,
    ksi_from_disp_fit,
)


def test_en_fit_lookup_keys():
    fits = [
        {"ch_idx": 0, "mom": 1, "fit": "fit_a"},
        {"ch_idx": 1, "mom": 0, "fit": "fit_b"},
    ]
    lookup = en_fit_lookup(fits)
    assert lookup[(0, 1)] == "fit_a"
    assert lookup[(1, 0)] == "fit_b"
    assert (0, 0) not in lookup


def test_disp_fit_lookup_by_channel():
    fits = [{"ch_idx": 2, "fit": "disp_fit"}]
    assert disp_fit_lookup(fits)[2] == "disp_fit"


def test_ksi_from_disp_fit():
    disp_fit = SimpleNamespace(p={"a": 0.25})
    at_invs = 7.219
    ns = 16
    expected = 2 * np.pi * at_invs / np.sqrt(0.25) / ns
    assert ksi_from_disp_fit(disp_fit, at_invs, ns) == expected
