from dataclasses import replace

from input.config import BuildConfig
from analysis.scattering import fit_mom_indices


def test_fit_mom_indices_rest_only():
    config = BuildConfig("X3872").build_config_from_control("tetraquark")
    config = replace(config, is_moving_frame=False, fit_mom_by_ns={16: [0, 1]})
    indices = fit_mom_indices(config, {})
    assert indices == {16: [0, 1]}


def test_fit_mom_indices_moving_frame_offsets_mf_levels():
    config = BuildConfig("X3872").build_config_from_control("tetraquark")
    config = replace(
        config,
        is_moving_frame=True,
        fit_mom_by_ns={16: [1]},
        fit_mom_by_ns_MF={16: [0, 2]},
    )
    scattering_dict = {"rest_point_count": {16: 3}}
    indices = fit_mom_indices(config, scattering_dict)
    assert indices[16] == [1, 3, 5]
