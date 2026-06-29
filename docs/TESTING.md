# Testing

How to run tests without lattice `.npy` data.

---

## Quick start

```bash
pip install -r requirements-dev.txt
MPLBACKEND=Agg pytest          # same as GitHub Actions CI
pytest -v                      # verbose
pytest tests/test_channels.py -v
```

CI runs on every push/PR to `main` (Python 3.10 & 3.12). See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## Principle

Tests use synthetic arrays and config-only checks. They verify shapes, formulas, file I/O, and **channel label → index** resolution — not agreement with published physics numbers.

---

## Test files

| File | Module under test | Technique |
|------|-------------------|-----------|
| `tests/conftest.py` | shared | `tcccc_tetra` / `tcccc_meson` fixtures; `assert_scattering_labels`, `tetra_key` |
| `tests/test_channels.py` | `input/config.py` | `scattering_channel` and ratio labels → indices (all three systems) |
| `tests/test_jackknife.py` | `statistics/jackknife.py` | Analytic mean/SE; leave-one-out slice |
| `tests/test_io.py` | `data/io.py`, `input/config.py` tags | `tmp_path` for `.npy` round-trip; path tag strings |
| `tests/test_scattering.py` | `analysis/scattering.py` | Random energies + constant zeta table |
| `tests/test_scattering_fit.py` | `analysis/scattering.py` (`fit_mom_indices`) | `BuildConfig` + `dataclasses.replace` (no `.npy`) |
| `tests/test_fit_mass.py` | `analysis/fit_mass.py` helpers | Lookup dicts; dispersion slope → ξ |
| `tests/test_fit_tmin.py` | `analysis/fit_tmin.py` | Ratio series; `ratio_scan_lookup`; E2_mom0/mom1 mapping |
| `tests/test_plot_set.py` | `plotting/plot_set.py` | Pure helpers (axis limits, GeV scale, labels) |
| `tests/test_analysis_switches.py` | `input/config.py` | Branch gating; `is_plot_title` / `is_plot_show` |

Shared fixture in `tests/conftest.py`:

```python
@pytest.fixture
def tcccc_tetra() -> Config:
    return BuildConfig("Tcccc6600").build_config_from_control("tetraquark")

def tetra_key(config, label: str, mom: int) -> tuple[int, int]:
    return ratio_point_by_label(config, label, mom).key
```

---

## Examples

### Channel label resolution (`test_channels.py`)

```python
match = resolve_scattering_chan(
    config.scattering_chan,
    config.meson_chan_name_list,
    config.tetra_chan_name_list,
)
assert config.meson_chan_name_list[match.meson_a] == r"J/\psi"
assert config.tetra_chan_name_list[match.tetra] == r"J/\psi\,J/\psi"
```

### Ratio lookup by label (`test_fit_tmin.py`)

```python
lookup = ratio_scan_lookup(tcccc_tetra)
key = ratio_point_by_label(tcccc_tetra, r"J/\psi\,J/\psi", 0).key
assert lookup[key].meson_chan == meson_idx(tcccc_tetra, r"J/\psi")
```

### Config-only test (`test_scattering_fit.py`)

```python
config = BuildConfig("X3872").build_config_from_control("tetraquark")
config = replace(config, run_MF_analysis=True, scattering_Ns_mom_MF={16: [0, 2]})
indices = fit_mom_indices(config, {})
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
- Full `lsqfit` effective-mass fits on real correlators (t_min scan fits)
- Plot pixel comparison against reference figures
- Moving-frame zeta table generation (`joblib` parallel sum)

Validate those locally with real data under `data/<system>/` after `pytest` passes.

---

## Adding tests

1. Prefer testing **one public function** with minimal synthetic input.
2. Use `tmp_path` for any `.npy` read/write.
3. Use `BuildConfig("Tcccc6600")` and **label fields** (`scattering_channel`, `ratio_point_by_label`) rather than hard-coded chan indices when possible.
4. Keep tests independent of `data/**/*.npy` so CI stays fast.
