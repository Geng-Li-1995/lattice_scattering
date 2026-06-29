"""Ratio plot y-axis limits use the fit window only."""

import numpy as np

from plotting.plot_ratio import RatioPlotter, _RatioChanPlot


def _make_chan(*, t_min: int, half_Nt: int, plateau: float, spike: float) -> _RatioChanPlot:
    t_all = np.arange(half_Nt)
    ratio_mean = np.full(half_Nt, plateau)
    ratio_mean[:t_min] = spike
    ratio_err = np.full(half_Nt, 0.01)
    t_fit = np.arange(t_min, half_Nt)
    return _RatioChanPlot(
        chan_idx=0,
        mom=0,
        t_min=t_min,
        t_fit=t_fit,
        t_all=t_all,
        ratio_mean=ratio_mean,
        ratio_err=ratio_err,
        finite=np.isfinite(ratio_mean),
        y_fit_m=np.full(len(t_fit), plateau),
        y_fit_s=np.full(len(t_fit), 0.005),
        ref_fit=None,
        is_ratio_shift=True,
    )


def test_ratio_ylim_ignores_pre_tmin_spikes():
    """L12-style: early-time ratio spikes must not set y-axis range."""
    half_Nt = 48
    chan = _make_chan(t_min=24, half_Nt=half_Nt, plateau=0.2, spike=80.0)
    vals = RatioPlotter._chan_y_values(chan, half_Nt)
    assert np.nanmax(vals) < 1.0
    assert np.nanmin(vals) > 0.0
