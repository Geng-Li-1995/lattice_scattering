import numpy as np

from plotting.plot_set import axis_limits_from_values


def test_axis_limits_from_values_adds_padding():
    lo, hi = axis_limits_from_values([0.0, 10.0], padding_fraction=0.10)
    assert lo == -1.0
    assert hi == 11.0


def test_axis_limits_single_point_uses_fallback_span():
    lo, hi = axis_limits_from_values([5.0], padding_fraction=0.10)
    assert lo < 5.0
    assert hi > 5.0
