# Testing

How to run tests without lattice `.npy` data.

---

## Quick start

```bash
pip install -r requirements-dev.txt
MPLBACKEND=Agg pytest
```

CI runs on Python 3.10 & 3.12 ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

---

## Principle

Synthetic arrays and config-only checks — no `data/**/*.npy` in CI.

---

## Test files (~19 cases)

| File | Covers |
|------|--------|
| `test_config.py` | Channel labels (3 systems), ratio lookup, branch/plot switches |
| `test_fit_tmin.py` | Ratio series, t_run, fit + jackknife errors |
| `test_fit_mass.py` | Dispersion → ξ |
| `test_io.py` | Path tags, MF scatter round-trip |
| `test_jackknife.py` | Mean, error, resample |
| `test_scattering.py` | Lüscher algebra, `fit_mom_indices` |
| `test_plot.py` | `plot_set` helpers, ratio y-limits |

Fixtures: `conftest.py` (`tcccc_tetra`, `x3872_tetra`, `zc3900_tetra`, `rng`).

---

## Not tested

End-to-end `main.py`, GEVP on real 6D data, plot pixel comparison, zeta table generation.

Validate those locally after `pytest` passes.
