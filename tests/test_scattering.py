"""Scattering algebra and momentum-index mapping."""

from dataclasses import replace

import numpy as np

from analysis.scattering import _analyze_rest_momentum, fit_mom_indices, q_sq_linspace
from input.config import BuildConfig, scattering_momenta_from_db


def test_q_sq_linspace_endpoints():
    grid = q_sq_linspace(5)
    assert grid.shape == (5,) and grid[0] == -5.0 and grid[-1] == 10.0


def test_analyze_rest_momentum(rng):
    n = 20
    en = 3.9 + 0.001 * rng.standard_normal(n)
    ma = 1.897 + 0.001 * rng.standard_normal(n)
    mb = 2.018 + 0.001 * rng.standard_normal(n)
    lattice_size = np.full(n, 11.1) + 0.01 * rng.standard_normal(n)
    zeta = np.full(50, 0.4)

    result = _analyze_rest_momentum(en, ma, mb, lattice_size, q_sq_linspace(50), zeta)
    assert set(result) == {"Ks", "s", "sqrt_s", "k_sq", "kcot"}
    expected = 0.4 * (2.0 / np.sqrt(np.pi)) / lattice_size
    assert np.allclose(result["kcot"], expected, rtol=1e-12)


def test_scattering_momentum_indices():
    builder = BuildConfig("Tcccc6600")
    key = (12, 96, 420, 170)
    assert scattering_momenta_from_db(builder.ensemble_db, key, r"J/\psi\,J/\psi") == [
        0, 1, 2, 3, 4
    ]
    assert [k[0] for k in builder.input_control.scattering_list] == [12, 16]

    config = builder.build_config_from_control("tetraquark")
    plot_moms = {12: {"rest": [0, 1, 2, 3, 4]}, 16: {"rest": [0, 1, 2, 3, 4]}}
    assert fit_mom_indices(config, {"plot_moms": plot_moms}) == {12: [0, 1, 2], 16: [0, 1]}

    x3872 = BuildConfig("X3872").build_config_from_control("tetraquark")
    x3872 = replace(
        x3872,
        run_MF_analysis=True,
        scattering_Ns_mom={16: [1]},
        scattering_Ns_mom_MF={16: [0, 2]},
    )
    assert fit_mom_indices(x3872, {"plot_moms": {16: {"rest": [0, 1, 2], "MF": 3}}})[16] == [
        1, 3, 5
    ]
