import gvar as gv
import numpy as np
from dataclasses import replace

from analysis.fit_tmin import (
    RatioPair,
    build_ratio_series,
    build_ratio_series_distinguishable,
    fit_ratio_reference,
    mesons_are_identical,
    ratio_delta_prior_value,
    ratio_scan_lookup,
    ratio_series_mean_err,
    resolve_ratio_pair,
    t_run_list,
)
from input.config import BuildConfig

from tests.conftest import meson_idx, tetra_key


def test_build_ratio_series_shape_and_shift():
    nt = 128
    half = nt // 2
    n_sample = 5
    tetra = np.ones((nt, n_sample))
    meson = np.linspace(1.0, 2.0, nt)[:, None] * np.ones((nt, n_sample))
    ratio_at = 1
    ratio = build_ratio_series(tetra, meson, ratio_at, half)
    assert ratio.shape == (half, n_sample)


def test_ratio_at_matches_legacy_ratio_ta_two():
    """ratio_at=1 with step 2*ratio_at equals old ratio_ta=2 implementation."""
    nt = 64
    half = nt // 2
    n_sample = 4
    rng = np.random.default_rng(1)
    tetra = np.exp(-0.3 * np.arange(nt)[:, None]) * (1 + 0.02 * rng.standard_normal((nt, n_sample)))
    meson = np.exp(-0.2 * np.arange(nt)[:, None]) * (1 + 0.02 * rng.standard_normal((nt, n_sample)))

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
    current = build_ratio_series(tetra, meson, ratio_at=1, half_Nt=half)
    np.testing.assert_allclose(current, legacy, rtol=0, atol=0, equal_nan=True)


def test_build_ratio_series_distinguishable_shape():
    nt = 128
    half = nt // 2
    n_sample = 4
    tetra = np.linspace(2.0, 4.0, nt)[:, None] * np.ones((nt, n_sample))
    meson_a = np.linspace(1.0, 1.5, nt)[:, None] * np.ones((nt, n_sample))
    meson_b = np.linspace(0.8, 1.2, nt)[:, None] * np.ones((nt, n_sample))
    ratio = build_ratio_series_distinguishable(tetra, meson_a, meson_b, half)
    assert ratio.shape == (half, n_sample)


def test_build_ratio_series_zero_denominator_is_nan():
    nt = 32
    half = nt // 2
    n_sample = 3
    ratio_at = 1
    meson = np.ones((nt, n_sample))
    tetra = np.ones((nt, n_sample))
    ratio = build_ratio_series(tetra, meson, ratio_at, half)
    assert np.isnan(ratio).any()
    assert not np.isinf(ratio).any()


def test_meson_tmin_uses_two_state_config():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(builder.input_control, run_tmin_analysis=True)
    meson = builder.build_config_from_control("meson")
    tetra = builder.build_config_from_control("tetraquark")
    assert meson.n_state == 2
    assert meson.run_tmin_analysis is True
    assert meson.run_ratio_analysis is False
    assert tetra.n_state == 3
    assert tetra.run_tmin_analysis is True


def test_t_run_list_defaults():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    t_list = t_run_list(config)
    assert t_list[0] == 10
    assert t_list[-1] == config.lattice_Nt // 2 - config.t_run_end_offset - 1


def test_ratio_delta_prior_uses_tetra_meff_prior():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    val = ratio_delta_prior_value(config, 0, 1)
    assert val == config.meff_prior[0][1][0]
    assert val > 0.1


def test_resolve_ratio_pair_identical_mesons(tcccc_tetra):
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")
    key = tetra_key(tcccc_tetra, r"J/\psi\,J/\psi", 2)
    pair = resolve_ratio_pair(key, tcccc_tetra)
    assert pair == RatioPair(1, 2, jpsi, 2, jpsi, 2, True)
    assert resolve_ratio_pair((1, 2, 0, 3), tcccc_tetra) == RatioPair(
        1, 2, 0, 3, 0, 3, True
    )


def test_resolve_ratio_pair_distinguishable_mesons(x3872_tetra):
    assert x3872_tetra.is_ratio_shift is False
    d = meson_idx(x3872_tetra, r"D")
    dstar = meson_idx(x3872_tetra, r"D^*")
    key = tetra_key(x3872_tetra, r"D\,D^*", 1)
    assert resolve_ratio_pair(key, x3872_tetra) == RatioPair(
        1, 1, d, 1, dstar, 1, False
    )


def test_resolve_ratio_pair_from_tetra_name_zc3900(zc3900_tetra):
    pi = meson_idx(zc3900_tetra, r"\pi")
    jpsi = meson_idx(zc3900_tetra, r"J/\psi")
    d = meson_idx(zc3900_tetra, r"D")
    dstar = meson_idx(zc3900_tetra, r"D^*")
    assert resolve_ratio_pair(
        tetra_key(zc3900_tetra, r"\pi\,J/\psi", 1), zc3900_tetra
    ) == RatioPair(0, 1, pi, 1, jpsi, 1, False)
    assert resolve_ratio_pair(
        tetra_key(zc3900_tetra, r"D\,D^*", 0), zc3900_tetra
    ) == RatioPair(2, 0, d, 0, dstar, 0, False)


def test_mesons_are_identical():
    assert mesons_are_identical(1, 2, 1, 2)
    assert not mesons_are_identical(0, 1, 1, 1)


def test_ratio_scan_lookup_maps_tetra_indices(tcccc_tetra):
    lookup = ratio_scan_lookup(tcccc_tetra)
    assert lookup, "Tcccc6600 run_ratio_analysis should scan all tetraquark chans"
    target = lookup[tetra_key(tcccc_tetra, r"J/\psi\,J/\psi", 0)]
    assert target.is_ratio_shift
    assert target.meson_chan == meson_idx(tcccc_tetra, r"J/\psi")
    assert target.tetra_mom == target.meson_mom == 0
    assert target.delta_prior == tcccc_tetra.meff_prior[target.tetra_chan][target.tetra_mom][0]


def test_fit_ratio_reference_identical_meson():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    meson_config = BuildConfig("Tcccc6600").build_config_from_control("meson")
    lookup = ratio_scan_lookup(config)
    target = lookup[(1, 0)]
    nt = config.lattice_Nt
    half = nt // 2
    n_sample = 8
    rng = np.random.default_rng(0)
    meson = np.exp(-0.42 * np.arange(nt)[:, None]) * (1 + 0.01 * rng.standard_normal((nt, n_sample)))
    tetra = np.exp(-0.86 * np.arange(nt)[:, None]) * (1 + 0.01 * rng.standard_normal((nt, n_sample)))
    ref = fit_ratio_reference(config, tetra, meson, meson, target, meson_config)
    assert ref.target.is_ratio_shift
    assert ref.ratio_data.shape == (half, n_sample)
    assert float(gv.mean(ref.total_energy)) > 0


def test_ratio_series_mean_err_matches_cosh_workflow(rng):
    """Ratio errors use LOO correlators → R(t) → jackknife, same path as plot_En cosh."""
    from analysis.models import MathModels
    from statistics.resample import jackknife_map_gvar

    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    nt = config.lattice_Nt
    n_sample = 20
    meson = np.exp(-0.42 * np.arange(nt)[:, None]) * (
        1 + 0.01 * rng.standard_normal((nt, n_sample))
    )
    tetra = np.exp(-0.86 * np.arange(nt)[:, None]) * (
        1 + 0.01 * rng.standard_normal((nt, n_sample))
    )
    t = 30

    mean_r, err_r = ratio_series_mean_err(
        config, tetra, meson, meson, is_ratio_shift=True
    )
    m_en, e_en = jackknife_map_gvar(
        config,
        tetra,
        lambda jack: MathModels.generate_cosh_from_data(jack, time_axis=0),
    )

    rel_r = err_r[t] / abs(mean_r[t])
    rel_en = e_en[t] / m_en[t]
    assert 0.0001 < rel_r < 0.01
    assert 0.2 < rel_r / rel_en < 5.0, f"ratio/en error ratio {rel_r/rel_en:.2f}"


def test_ratio_series_mean_err_length():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    nt = config.lattice_Nt
    half = nt // 2
    n_sample = 6
    tetra = np.ones((nt, n_sample))
    meson = np.linspace(1.0, 2.0, nt)[:, None] * np.ones((nt, n_sample))
    mean, err = ratio_series_mean_err(
        config, tetra, meson, meson, is_ratio_shift=True
    )
    assert mean.shape == (half,)
    assert err.shape == (half,)


def test_e2_mom0_mom1_ratio_maps_to_jpsi_meson(tcccc_tetra, tcccc_meson):
    """E2_mom0 / E2_mom1: J/psi J/psi at n^2=0,1 uses meson J/psi at same n^2."""
    lookup = ratio_scan_lookup(tcccc_tetra)
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")

    mom0 = lookup[tetra_key(tcccc_tetra, r"J/\psi\,J/\psi", 0)]
    assert mom0.tetra_mom == mom0.meson_mom == 0
    assert mom0.meson_chan == jpsi
    assert mom0.is_ratio_shift
    assert mom0.state_idx == 3

    mom1 = lookup[tetra_key(tcccc_tetra, r"J/\psi\,J/\psi", 1)]
    assert mom1.tetra_mom == mom1.meson_mom == 1
    assert mom1.meson_chan == jpsi
    assert mom1.state_idx == 4

    assert tcccc_meson.chan_name_list[jpsi] == r"J/\psi"
    assert 0 in tcccc_meson.chan_momentum_list[jpsi]
    assert 1 in tcccc_meson.chan_momentum_list[jpsi]
