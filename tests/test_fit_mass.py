from types import SimpleNamespace

import numpy as np

from analysis.fit_mass import ksi_from_disp_fit


def test_ksi_from_disp_fit():
    disp_fit = SimpleNamespace(p={"a": 0.25})
    at_invs = 7.219
    Ns = 16
    expected = 2 * np.pi * at_invs / np.sqrt(0.25) / Ns
    assert ksi_from_disp_fit(disp_fit, at_invs, Ns) == expected
