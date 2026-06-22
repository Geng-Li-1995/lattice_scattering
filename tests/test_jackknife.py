import numpy as np
import pytest

from statistics.jackknife import Jackknife


def test_jackknife_mean_matches_sample_mean():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    jk = Jackknife(data)
    assert jk.mean() == pytest.approx(np.mean(data))


def test_jackknife_error_matches_analytic_mean_case():
    n = 401
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    loo_means = np.array([(x.sum() - x[i]) / (n - 1) for i in range(n)])

    jk = Jackknife(loo_means)
    expected_se = x.std(ddof=1) / np.sqrt(n)
    assert jk.error() == pytest.approx(expected_se, rel=1e-10)


def test_jackknife_resample_manual_removes_one_sample():
    data = np.arange(5, dtype=float)
    jk = Jackknife(data)
    assert np.array_equal(jk.resample_manual(2), np.array([0.0, 1.0, 3.0, 4.0]))


def test_jackknife_error_requires_at_least_two_samples():
    with pytest.raises(ValueError, match="at least 2 samples"):
        Jackknife(np.array([1.0])).error()
