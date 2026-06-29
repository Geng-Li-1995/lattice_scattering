"""Plot helpers (no figure rendering)."""

import numpy as np
import pytest

from plotting.plot_ratio import RatioPlotter, _RatioChanPlot
from plotting.plot_set import (
    axis_limits_from_values,
    chan_mom_label,
    combine_axis_limits,
    energy_gev_mean,
    y_limits_from_error,
)


def test_plot_set_helpers():
    assert chan_mom_label("J/psi", 2) == "J/psi(n^2=2)"
    lo, hi = y_limits_from_error(3.0, 0.01, scale=10.0)
    assert lo == pytest.approx(2.9) and hi == pytest.approx(3.1)
    lo, hi = combine_axis_limits(axis_limits_from_values([1.0, 2.0]), axis_limits_from_values([5.0, 6.0]))
    assert lo < 1.0 and hi > 6.0

    import gvar as gv

    np.testing.assert_allclose(energy_gev_mean(7.0, [gv.gvar(1.0, 0.1)]), [7.0])


def test_ratio_ylim_uses_fit_window_only():
    half_Nt, t_min, plateau, spike = 48, 24, 0.2, 80.0
    t_all = np.arange(half_Nt)
    ratio_mean = np.full(half_Nt, plateau)
    ratio_mean[:t_min] = spike
    chan = _RatioChanPlot(
        chan_idx=0, mom=0, t_min=t_min,
        t_fit=np.arange(t_min, half_Nt), t_all=t_all,
        ratio_mean=ratio_mean, ratio_err=np.full(half_Nt, 0.01),
        finite=np.isfinite(ratio_mean),
        y_fit_m=np.full(half_Nt - t_min, plateau),
        y_fit_s=np.full(half_Nt - t_min, 0.005),
        ref_fit=None, is_ratio_shift=True,
    )
    vals = RatioPlotter._chan_y_values(chan, half_Nt)
    assert np.nanmax(vals) < 1.0 and np.nanmin(vals) > 0.0
