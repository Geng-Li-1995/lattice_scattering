"""Channel resolution: label fields → indices (scattering, ratio, meson lookup)."""

from __future__ import annotations

import pytest
from dataclasses import replace

from analysis.fit_tmin import RatioPair, ratio_scan_lookup, resolve_ratio_pair
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

from tests.conftest import (
    assert_ratio_point_labels,
    assert_scattering_labels,
    meson_idx,
    tetra_key,
)


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


def test_normalize_chan_label_ignores_separators():
    assert normalize_chan_label(r"J/\psi\,J/\psi") == normalize_chan_label(r"J/\psi,J/\psi")


def test_split_tetra_chan_name():
    assert split_tetra_chan_name(r"\rho\,\eta_c") == [r"\rho", r"\eta_c"]


def test_chan_mom_by_label_meson():
    config = BuildConfig("Tcccc6600").build_config_from_control("meson")
    point = chan_mom_by_label(r"J/\psi", 2, config.chan_name_list)
    assert point == ChanMom(meson_idx(config, r"J/\psi"), 2)


def test_ratio_requires_tetra_chan_name_list(tcccc_tetra):
    empty = replace(tcccc_tetra, tetra_chan_name_list=[])
    with pytest.raises(ValueError, match="chan_name_list"):
        ratio_scan_points_for_config(empty)


def test_ratio_scan_points_match_iter_chan_mom(tcccc_tetra):
    expected = [p.key for p in iter_chan_mom_points(tcccc_tetra)]
    assert ratio_scan_points_for_config(tcccc_tetra) == expected


def test_resolve_mesons_from_tetra_label(tcccc_tetra):
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")
    tetra_chan = meson_chan_index(
        r"J/\psi\,J/\psi", tcccc_tetra.tetra_chan_name_list
    )
    assert resolve_mesons_from_tetra_chan(tetra_chan, 2, tcccc_tetra) == (jpsi, 2, jpsi, 2)


def test_resolve_mesons_missing_momentum_raises(tcccc_tetra):
    tetra_chan = meson_chan_index(
        r"J/\psi\,J/\psi", tcccc_tetra.tetra_chan_name_list
    )
    with pytest.raises(ValueError, match="n\\^2=7"):
        resolve_mesons_from_tetra_chan(tetra_chan, 7, tcccc_tetra)


def test_scattering_single_meson_part_raises(tcccc_tetra):
    with pytest.raises(ValueError, match="expected 2"):
        resolve_scattering_chan(
            r"\chi_{c1}",
            tcccc_tetra.meson_chan_name_list,
            [r"\chi_{c1}"],
        )


@pytest.mark.parametrize("mom", [0, 1])
def test_ratio_point_by_label_tcccc(tcccc_tetra, mom):
    label = r"J/\psi\,J/\psi"
    point = ratio_point_by_label(tcccc_tetra, label, mom)
    assert_ratio_point_labels(
        tcccc_tetra, point, tetra=label, meson_a=r"J/\psi", meson_b=r"J/\psi"
    )
    assert ratio_scan_lookup(tcccc_tetra)[point.key].is_ratio_shift


@pytest.mark.parametrize(
    "config_fixture, label, mom, meson_a, meson_b",
    [
        ("x3872_tetra", r"D\,D^*", 1, r"D", r"D^*"),
        ("zc3900_tetra", r"\pi\,J/\psi", 1, r"\pi", r"J/\psi"),
    ],
)
def test_ratio_pair_by_label(request, config_fixture, label, mom, meson_a, meson_b):
    config = request.getfixturevalue(config_fixture)
    pair = resolve_ratio_pair(ratio_point_by_label(config, label, mom).key, config)
    meson_names = config.meson_chan_name_list
    assert meson_names[pair.meson_a_chan] == meson_a
    assert meson_names[pair.meson_b_chan] == meson_b
    assert not pair.is_ratio_shift


def test_resolve_ratio_pair_returns_ratio_pair(tcccc_tetra):
    pair = resolve_ratio_pair(tetra_key(tcccc_tetra, r"J/\psi\,J/\psi", 2), tcccc_tetra)
    assert isinstance(pair, RatioPair)
    jpsi = meson_idx(tcccc_tetra, r"J/\psi")
    assert pair == RatioPair(1, 2, jpsi, 2, jpsi, 2, True)


def test_ratio_scan_lookup_keys_from_labels(tcccc_tetra):
    lookup = ratio_scan_lookup(tcccc_tetra)
    eta_c = r"\eta_c\,\eta_c"
    jpsi = r"J/\psi\,J/\psi"
    assert set(lookup) == {
        tetra_key(tcccc_tetra, eta_c, 1),
        tetra_key(tcccc_tetra, eta_c, 2),
        tetra_key(tcccc_tetra, eta_c, 4),
        *(tetra_key(tcccc_tetra, jpsi, m) for m in range(5)),
    }
