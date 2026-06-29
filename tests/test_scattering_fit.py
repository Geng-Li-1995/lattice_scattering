from dataclasses import replace

from analysis.scattering import fit_mom_indices
from input.config import BuildConfig, scattering_momenta_from_db


def test_scattering_momenta_from_db_tcccc6600():
    builder = BuildConfig("Tcccc6600")
    key = (12, 96, 420, 170)
    moms = scattering_momenta_from_db(
        builder.ensemble_db, key, r"J/\psi\,J/\psi"
    )
    assert moms == [0, 1, 2, 3, 4]


def test_fit_mom_indices_maps_fit_subset_to_plot_rows():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    scattering_dict = {
        "plot_moms": {
            12: {"rest": [0, 1, 2, 3, 4]},
            16: {"rest": [0, 1, 2, 3, 4]},
        }
    }
    indices = fit_mom_indices(config, scattering_dict)
    assert indices[12] == [0, 1, 2]
    assert indices[16] == [0, 1]


def test_fit_mom_indices_MF():
    config = BuildConfig("X3872").build_config_from_control("tetraquark")
    config = replace(
        config,
        run_MF_analysis=True,
        scattering_Ns_mom={16: [1]},
        scattering_Ns_mom_MF={16: [0, 2]},
    )
    scattering_dict = {
        "plot_moms": {16: {"rest": [0, 1, 2], "MF": 3}},
    }
    assert fit_mom_indices(config, scattering_dict)[16] == [1, 3, 5]


def test_scattering_list_from_scattering_Ns_mom():
    ctrl = BuildConfig("Tcccc6600").input_control
    assert [key[0] for key in ctrl.scattering_list] == [12, 16]
