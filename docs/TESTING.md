# Testing

How to run tests without lattice `.npy` data.

---

## Quick start

```bash
pip install -r requirements-dev.txt
MPLBACKEND=Agg pytest          # same as GitHub Actions CI
pytest -v                      # verbose
pytest tests/test_scattering.py -v
```

CI runs on every push/PR to `main` (Python 3.10 & 3.12). See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## Principle

Tests use synthetic arrays and config-only checks. They verify shapes, formulas, and file I/O — not agreement with published physics numbers.

---

## Test files

| File | Module under test | Technique |
|------|-------------------|-----------|
| `tests/test_jackknife.py` | `statistics/jackknife.py` | Analytic mean/SE; leave-one-out slice |
| `tests/test_io.py` | `data/io.py`, `input/config.py` tags | `tmp_path` for `.npy` round-trip; path tag strings |
| `tests/test_scattering.py` | `analysis/scattering.py` | Random energies + constant zeta table |
| `tests/test_scattering_fit.py` | `analysis/scattering.py` (`fit_mom_indices`) | `BuildConfig` + `dataclasses.replace` (no `.npy`) |
| `tests/test_fit_mass.py` | `analysis/fit_mass.py` helpers | Lookup dicts; dispersion slope → ξ |
| `tests/test_fit_tmin.py` | `analysis/fit_tmin.py` | Ratio series; `ratio_scan_lookup`; E2_mom0/mom1 mapping |
| `tests/test_plot_tmin.py` | `plotting/plot_tmin.py` | GeV y-axis limits (±10× reference error) |
| `tests/test_plot_set.py` | `plotting/plot_set.py` | Pure axis-limit math |

Shared fixture in `tests/conftest.py`:

```python
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)   # reproducible noise
```

---

## Examples

### Synthetic scattering sample (`test_scattering.py`)

```python
en = 3.9 + 0.001 * rng.standard_normal(20)   # fake tetraquark energies
zeta_00_rest = np.full_like(q_sq_grid, 0.4)    # flat zeta table
result = _analyze_rest_momentum(en, ma, mb, lattice_size, q_sq_grid, zeta_00_rest)
assert result["kcot"].shape == (20,)
```

### Config-only test (`test_scattering_fit.py`)

```python
config = BuildConfig("X3872").build_config_from_control("tetraquark")
config = replace(config, is_moving_frame=True, fit_mom_by_ns_MF={16: [0, 2]})
indices = fit_mom_indices(config, {"rest_point_count": {16: 3}})
assert indices[16] == [1, 3, 5]
```

Only Python config in `input/input_X3872.py` is loaded — no correlator files.

### Ratio channel mapping (`test_fit_tmin.py`)

```python
lookup = ratio_scan_lookup(tetra_config)
assert lookup[(1, 0)].meson_ch == 1 and lookup[(1, 0)].meson_mom == 0  # E2_mom0 → J/psi
assert lookup[(1, 1)].meson_ch == 1 and lookup[(1, 1)].meson_mom == 1
```

### Temporary file I/O (`test_io.py`)

```python
def test_mf_scatter_file_roundtrip(tmp_path: Path):
    path = tmp_path / "scatter.npy"
    save_mf_scatter_all(path, results)
    loaded = load_mf_scatter_all(path)
```

`tmp_path` is an empty directory pytest creates and deletes automatically.

---

## What is **not** tested

- End-to-end `main.py` workflow
- GEVP diagonalization on real tetraquark tensors
- Full `lsqfit` effective-mass fits on real correlators (t_min scan fits)
- Plot pixel comparison against reference figures
- Moving-frame zeta table generation (`joblib` parallel sum)

Validate those locally with real data under `data/<system>/` after `pytest` passes.

---

## Adding tests

1. Prefer testing **one public function** with minimal synthetic input.
2. Use `tmp_path` for any `.npy` read/write.
3. Use `BuildConfig("Tcccc6600")` when only `ENSEMBLE_DB` / `InputControl` fields matter.
4. Keep tests independent of `data/**/*.npy` so CI stays fast.
