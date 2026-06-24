# Testing Guide

How the test suite validates the pipeline **without** lattice `.npy` data.

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

## Design principle

| Full pipeline (`main.py`) | Unit tests (`pytest`) |
|---------------------------|------------------------|
| Reads multi‑MiB `data/**/raw/*.npy` | Uses **synthetic NumPy arrays** |
| GEVP + `lsqfit` on real correlators | Tests **single functions** in isolation |
| Jackknife over 400 configurations | Jackknife on 4–401 fake numbers |
| Writes `result/` figures | Plot helpers tested without saving figures |

Tests check **logic** (shapes, keys, formulas, file round-trips), not physics agreement with published numbers.

---

## Test files

| File | Module under test | Technique |
|------|-------------------|-----------|
| `tests/test_jackknife.py` | `statistics/jackknife.py` | Analytic mean/SE; leave-one-out slice |
| `tests/test_io.py` | `data/io.py`, `input/config.py` tags | `tmp_path` for `.npy` round-trip; path tag strings |
| `tests/test_scattering.py` | `analysis/scattering.py` | Random energies + constant zeta table |
| `tests/test_scattering_fit.py` | `analysis/scattering.py` (`fit_mom_indices`) | `BuildConfig` + `dataclasses.replace` (no `.npy`) |
| `tests/test_fit_mass.py` | `analysis/fit_mass.py` helpers | Lookup dicts; dispersion slope → ξ |
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
- Full `lsqfit` effective-mass fits
- Plot pixel comparison against reference figures
- Moving-frame zeta table generation (`joblib` parallel sum)

Validate those locally with real data under `data/<system>/` after `pytest` passes.

---

## Adding tests

1. Prefer testing **one public function** with minimal synthetic input.
2. Use `tmp_path` for any `.npy` read/write.
3. Use `BuildConfig("Tcccc6600")` when only `ENSEMBLE_DB` / `InputControl` fields matter.
4. Keep tests independent of `data/**/*.npy` so CI stays fast.
