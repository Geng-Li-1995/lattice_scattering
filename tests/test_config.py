"""Config: channel labels, branch switches, plot flags."""

from __future__ import annotations

import pytest
from dataclasses import replace

from analysis.fit_tmin import RatioPair, ratio_scan_lookup, resolve_ratio_pair
from analysis.scattering import run_scattering_analysis
from input.config import (
    BuildConfig,
    ChanMom,
    chan_mom_by_label,
    iter_chan_mom_points,
    meson_chan_index,
    normalize_chan_label,
    ratio_point_by_label,
    ratio_scan_points_for_config,
    resolve_mesons_from_tetra_chan,
    resolve_scattering_chan,
    split_tetra_chan_name,
)

from tests.conftest import assert_scattering_labels, meson_idx, tetra_key


@pytest.mark.parametrize(
    "system, meson_a, meson_b, tetra",
    [
        ("Tcccc6600", r"J/\psi", r"J/\psi", r"J/\psi\,J/\psi"),
        ("Zc3900", r"\rho", r"\eta_c", r"\rho\,\eta_c"),
        ("X3872", r"D", r"D^*", r"D\,D^*"),
    ],
)
def test_scattering_channel_by_label(system, meson_a, meson_b, tetra):
    config = BuildConfig(system).build_config_from_control("tetraquark")
    match = resolve_scattering_chan(
        config.scattering_chan,
        config.meson_chan_name_list,
        config.tetra_chan_name_list,
    )
    assert_scattering_labels(config, match, meson_a=meson_a, meson_b=meson_b, tetra=tetra)


def test_channel_label_helpers(tcccc_tetra):
    assert normalize_chan_label(r"J/\psi\,J/\psi") == normalize_chan_label(r"J/\psi,J/\psi")
    assert split_tetra_chan_name(r"\rho\,\eta_c") == [r"\rho", r"\eta_c"]

    meson = BuildConfig("Tcccc6600").build_config_from_control("meson")
    assert chan_mom_by_label(r"J/\psi", 2, meson.chan_name_list) == ChanMom(
        meson_idx(meson, r"J/\psi"), 2
    )

    jpsi_chan = meson_chan_index(r"J/\psi\,J/\psi", tcccc_tetra.tetra_chan_name_list)
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")
    assert resolve_mesons_from_tetra_chan(jpsi_chan, 2, tcccc_tetra) == (jpsi, 2, jpsi, 2)
    with pytest.raises(ValueError, match="n\\^2=7"):
        resolve_mesons_from_tetra_chan(jpsi_chan, 7, tcccc_tetra)
    with pytest.raises(ValueError, match="expected 2"):
        resolve_scattering_chan(
            r"\chi_{c1}", tcccc_tetra.meson_chan_name_list, [r"\chi_{c1}"]
        )


def test_ratio_scan_and_lookup(tcccc_tetra, x3872_tetra, zc3900_tetra):
    assert ratio_scan_points_for_config(tcccc_tetra) == [
        p.key for p in iter_chan_mom_points(tcccc_tetra)
    ]
    with pytest.raises(ValueError, match="chan_name_list"):
        ratio_scan_points_for_config(replace(tcccc_tetra, tetra_chan_name_list=[]))

    lookup = ratio_scan_lookup(tcccc_tetra)
    jpsi_label = r"J/\psi\,J/\psi"
    eta_c_label = r"\eta_c\,\eta_c"
    assert set(lookup) == {
        tetra_key(tcccc_tetra, eta_c_label, m) for m in (1, 2, 4)
    } | {tetra_key(tcccc_tetra, jpsi_label, m) for m in range(5)}

    mom0 = lookup[tetra_key(tcccc_tetra, jpsi_label, 0)]
    assert mom0.is_ratio_shift and mom0.meson_chan == meson_idx(tcccc_tetra, r"J/\psi")
    mom1 = lookup[tetra_key(tcccc_tetra, jpsi_label, 1)]
    assert mom1.state_idx == mom0.state_idx + 1

    pair = resolve_ratio_pair(tetra_key(tcccc_tetra, jpsi_label, 2), tcccc_tetra)
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")
    assert pair == RatioPair(1, 2, jpsi, 2, jpsi, 2, True)

    d, dstar = meson_idx(x3872_tetra, r"D"), meson_idx(x3872_tetra, r"D^*")
    assert resolve_ratio_pair(
        tetra_key(x3872_tetra, r"D\,D^*", 1), x3872_tetra
    ) == RatioPair(1, 1, d, 1, dstar, 1, False)

    pi, jpsi_m = meson_idx(zc3900_tetra, r"\pi"), meson_idx(zc3900_tetra, r"J/\psi")
    assert resolve_ratio_pair(
        tetra_key(zc3900_tetra, r"\pi\,J/\psi", 1), zc3900_tetra
    ) == RatioPair(0, 1, pi, 1, jpsi_m, 1, False)


def test_branch_and_plot_switches():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(
        builder.input_control,
        run_dispersion_analysis=True,
        run_weight_analysis=True,
        run_GEVP_analysis=True,
        is_plot_title=False,
        is_plot_show=False,
    )
    tetra = builder.build_config_from_control("tetraquark")
    meson = builder.build_config_from_control("meson")
    assert not tetra.run_dispersion_analysis and not tetra.run_weight_analysis
    assert not meson.run_GEVP_analysis
    assert not tetra.is_plot_title and not tetra.is_plot_show

    tetra = replace(tetra, run_tetraquark_analysis=False, run_scattering_analysis=True)
    with pytest.raises(KeyError):
        run_scattering_analysis(tetra, {})
