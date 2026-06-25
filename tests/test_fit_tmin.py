import numpy as np

from analysis.fit_tmin import (
    build_ratio_series,
    ratio_delta_prior_value,
    ratio_scan_lookup,
    resolve_ratio_pair,
    t_run_list,
)
from input.config import BuildConfig


def test_build_ratio_series_shape_and_shift():
    nt = 128
    half = nt // 2
    n_sample = 5
    tetra = np.ones((nt, n_sample))
    meson = np.linspace(1.0, 2.0, nt)[:, None] * np.ones((nt, n_sample))
    ta = 2
    ratio = build_ratio_series(tetra, meson, ta, half)
    assert ratio.shape == (half, n_sample)


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


def test_resolve_ratio_pair_same_ch_mom_for_identical_particles():
    assert resolve_ratio_pair((1, 2)) == (1, 2, 1, 2)
    assert resolve_ratio_pair((1, 2, 0, 3)) == (1, 2, 0, 3)


def test_ratio_scan_lookup_maps_tetra_indices():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    lookup = ratio_scan_lookup(config)
    assert lookup, "Tcccc6600 should define ratio_scan_points when is_ratio"
    target = next(iter(lookup.values()))
    assert target.tetra_ch == target.meson_ch
    assert target.tetra_mom == target.meson_mom
    assert target.delta_prior == config.meff_prior[target.tetra_ch][target.tetra_mom][0]


def test_e2_mom0_mom1_ratio_maps_to_jpsi_meson():
    """E2_mom0 / E2_mom1 figures: tetra J/psi J/psi at n^2=0,1 uses meson J/psi C2 at same n^2."""
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    meson = BuildConfig("Tcccc6600").build_config_from_control("meson")
    lookup = ratio_scan_lookup(config)

    mom0 = lookup[(1, 0)]
    assert (mom0.tetra_ch, mom0.tetra_mom, mom0.meson_ch, mom0.meson_mom) == (1, 0, 1, 0)
    assert mom0.state_idx == 3  # fourth entry in ratio_scan_points

    mom1 = lookup[(1, 1)]
    assert (mom1.tetra_ch, mom1.tetra_mom, mom1.meson_ch, mom1.meson_mom) == (1, 1, 1, 1)
    assert mom1.state_idx == 4

    assert meson.chan_name_list[1] == r"J/\psi"
    assert 0 in meson.chan_momt_list[1]
    assert 1 in meson.chan_momt_list[1]
