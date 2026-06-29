import numpy as np

from analysis.scattering import _analyze_rest_momentum, q_sq_linspace


def test_q_sq_linspace_endpoints():
    grid = q_sq_linspace(5)
    assert grid.shape == (5,)
    assert grid[0] == -5.0
    assert grid[-1] == 10.0


def test_analyze_rest_momentum_shapes_and_keys(rng):
    n = 20
    en = 3.9 + 0.001 * rng.standard_normal(n)
    ma = 1.897 + 0.001 * rng.standard_normal(n)
    mb = 2.018 + 0.001 * rng.standard_normal(n)
    lattice_size = np.full(n, 11.1) + 0.01 * rng.standard_normal(n)
    q_sq_grid = q_sq_linspace(50)
    zeta_00_rest = np.full_like(q_sq_grid, 0.4)

    result = _analyze_rest_momentum(
        en, ma, mb, lattice_size, q_sq_grid, zeta_00_rest
    )

    assert set(result) == {"Ks", "s", "sqrt_s", "k_sq", "kcot"}
    for name in result:
        assert result[name].shape == (n,)

    kcot = result["kcot"]
    expected = 0.4 * (2.0 / np.sqrt(np.pi)) / lattice_size
    assert np.allclose(kcot, expected, rtol=1e-12)
