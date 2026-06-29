"""Unit tests for plotting/plot_set.py helpers (no matplotlib figures)."""

import numpy as np
import pytest

from plotting.plot_set import (
    axis_limits_from_values,
    chan_mom_label,
    chan_mom_latex,
    combine_axis_limits,
    energy_gev_mean,
    format_energy_gev,
    log_channel_tag,
    y_limits_from_error,
)


def test_chan_mom_label_and_latex():
    assert chan_mom_label("J/psi", 2) == "J/psi(n^2=2)"
    assert chan_mom_latex("J/psi", 2) == r"$J/psi(n^2=2)$"


def test_log_channel_tag():
    assert log_channel_tag(1, 0) == "channel 1 n^2=0"


def test_y_limits_from_error():
    lo, hi = y_limits_from_error(3.0, 0.01, scale=10.0)
    assert lo == pytest.approx(2.9)
    assert hi == pytest.approx(3.1)


def test_axis_limits_and_combine():
    lim_a = axis_limits_from_values([1.0, 2.0])
    lim_b = axis_limits_from_values([5.0, 6.0])
    lo, hi = combine_axis_limits(lim_a, lim_b)
    assert lo < 1.0
    assert hi > 6.0


def test_energy_gev_mean():
    import gvar as gv

    vals = [gv.gvar(1.0, 0.1), gv.gvar(2.0, 0.2)]
    out = energy_gev_mean(7.0, vals)
    np.testing.assert_allclose(out, [7.0, 14.0])


def test_format_energy_gev():
    text = format_energy_gev(3.3, 0.05)
    assert "GeV" in text
    assert "3.3" in text
