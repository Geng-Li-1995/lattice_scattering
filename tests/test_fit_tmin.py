"""Ratio series, fits, and jackknife errors (fit_tmin)."""

import gvar as gv
import numpy as np
from dataclasses import replace

from analysis.fit_tmin import (
    build_ratio_series,
    build_ratio_series_distinguishable,
    fit_ratio_reference,
    ratio_delta_prior_value,
    ratio_scan_lookup,
    ratio_series_mean_err,
    t_run_list,
)
from analysis.models import MathModels
from input.config import BuildConfig
from statistics.resample import jackknife_map_gvar


def test_ratio_series(rng):
    nt, half, n_sample = 64, 32, 4
    tetra = np.exp(-0.3 * np.arange(nt)[:, None]) * (
        1 + 0.02 * rng.standard_normal((nt, n_sample))
    )
    meson = np.exp(-0.2 * np.arange(nt)[:, None]) * (
        1 + 0.02 * rng.standard_normal((nt, n_sample))
    )
    meson_b = np.linspace(0.8, 1.2, nt)[:, None] * np.ones((nt, n_sample))

    assert build_ratio_series(tetra, meson, 1, half).shape == (half, n_sample)
    assert build_ratio_series_distinguishable(tetra, meson, meson_b, half).shape == (
        half,
        n_sample,
    )

    step = 2
    legacy = np.roll(
        np.divide(
            tetra - np.roll(tetra, -step, axis=0),
            meson**2 - np.roll(meson, -step, axis=0) ** 2,
            out=np.full_like(tetra, np.nan),
            where=(meson**2 - np.roll(meson, -step, axis=0) ** 2) != 0,
        ),
        1,
        axis=0,
    )[:half]
    np.testing.assert_allclose(
        build_ratio_series(tetra, meson, ratio_at=1, half_Nt=half),
        legacy,
        equal_nan=True,
    )

    flat = build_ratio_series(np.ones((nt, 3)), np.ones((nt, 3)), 1, half)
    assert np.isnan(flat).any() and not np.isinf(flat).any()


def test_ratio_config_and_t_run():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(builder.input_control, run_tmin_analysis=True)
    meson = builder.build_config_from_control("meson")
    tetra = builder.build_config_from_control("tetraquark")
    assert meson.n_state == 2 and tetra.n_state == 3
    assert meson.run_tmin_analysis and tetra.run_tmin_analysis
    assert not meson.run_ratio_analysis and tetra.run_ratio_analysis

    t_list = t_run_list(tetra)
    assert t_list[0] == 10
    assert t_list[-1] == tetra.lattice_Nt // 2 - tetra.t_run_end_offset - 1
    assert ratio_delta_prior_value(tetra, 0, 1) == tetra.meff_prior[0][1][0]


def test_ratio_fit_and_jackknife_errors(rng):
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    meson_config = BuildConfig("Tcccc6600").build_config_from_control("meson")
    target = ratio_scan_lookup(config)[(1, 0)]
    nt, half = config.lattice_Nt, config.lattice_Nt // 2

    fit_rng = np.random.default_rng(0)
    n_fit = 8
    meson_fit = np.exp(-0.42 * np.arange(nt)[:, None]) * (
        1 + 0.01 * fit_rng.standard_normal((nt, n_fit))
    )
    tetra_fit = np.exp(-0.86 * np.arange(nt)[:, None]) * (
        1 + 0.01 * fit_rng.standard_normal((nt, n_fit))
    )
    ref = fit_ratio_reference(config, tetra_fit, meson_fit, meson_fit, target, meson_config)
    assert ref.target.is_ratio_shift
    assert ref.ratio_data.shape == (half, n_fit)
    assert float(gv.mean(ref.total_energy)) > 0

    n_sample = 20
    meson = np.exp(-0.42 * np.arange(nt)[:, None]) * (
        1 + 0.01 * rng.standard_normal((nt, n_sample))
    )
    tetra = np.exp(-0.86 * np.arange(nt)[:, None]) * (
        1 + 0.01 * rng.standard_normal((nt, n_sample))
    )
    mean_r, err_r = ratio_series_mean_err(
        config, tetra, meson, meson, is_ratio_shift=True
    )
    assert mean_r.shape == (half,) == err_r.shape

    t = 30
    m_en, e_en = jackknife_map_gvar(
        config,
        tetra,
        lambda jack: MathModels.generate_cosh_from_data(jack, time_axis=0),
    )
    rel_r = err_r[t] / abs(mean_r[t])
    rel_en = e_en[t] / m_en[t]
    assert 0.0001 < rel_r < 0.01
    assert 0.2 < rel_r / rel_en < 5.0
