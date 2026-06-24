# Dependencies

Python package requirements and system-level dependencies for the lattice scattering pipeline.

---

## Python Version

**Python 3.10+** is required.

The codebase uses:

- `match` / `case` statements (`input/input_Tcccc6600.py`)
- PEP 604 union syntax `int | None` (`statistics/bootstrap.py`)

---

## Python Packages

Install all packages with:

```bash
pip install -r requirements.txt
```

| Package | Minimum Version | Role in this project |
|---------|-----------------|----------------------|
| **numpy** | 1.24 | Array storage, linear algebra, I/O for `.npy` correlators |
| **scipy** | 1.10 | `scipy.linalg.eig` for GEVP generalized eigenvalue problem |
| **matplotlib** | 3.7 | Figure output (PNG/PDF); LaTeX labels via `plotting/plot_set.py` (`RC_PARAMS`) |
| **gvar** | 11.0 | `gvar` objects for mean/error propagation in fits and plots |
| **lsqfit** | 12.0 | Bayesian nonlinear least-squares fitting (`lsf.nonlinear_fit`) |
| **joblib** | 1.2 | Parallel precomputation of Lüscher zeta function lookup table |

### Package relationships

```
main.py
  ├── numpy, scipy          → GEVP, array operations
  ├── gvar + lsqfit         → effective-mass & dispersion fits
  ├── matplotlib            → all plotting modules
  └── joblib                → zeta precomputation (scattering.py)
```

`gvar` and `lsqfit` are developed by the same author (C. Morningstar) and are the standard fitting stack in the lattice QCD community. They must be installed together.

### Install notes

- **PyPI**: All packages are available on PyPI.
  ```bash
  pip install numpy scipy matplotlib gvar lsqfit joblib
  ```
- **Conda** (alternative):
  ```bash
  conda install numpy scipy matplotlib joblib
  pip install gvar lsqfit    # gvar/lsqfit not on conda-forge by default
  ```

---

## System Dependencies

| Dependency | Required? | Purpose |
|------------|-----------|---------|
| **TeX distribution** | Yes (for default plots) | LaTeX rendering of physics symbols (\(\eta_c\), \(J/\psi\), etc.) |
| **dvipng** or **ghostscript** | Bundled with TeX | matplotlib LaTeX backend |

If TeX is not available, set `"text.usetex": False` in `plotting/plot_set.py` → `RC_PARAMS`, or install a minimal TeX subset.

---

## Standard Library (no install needed)

| Module | Used in |
|--------|---------|
| `dataclasses` | `input/config.py`, `input/input_*.py`, `data/correlators.py`, `analysis/gevp.py` |
| `typing` | Type hints across all modules |
| `importlib` | Dynamic loading of `input/input_<System>.py` in `BuildConfig` |
| `pathlib` | Data and zeta cache paths (`data/io.py`, `analysis/scattering.py`) |
| `collections` | `defaultdict` in `analysis/scattering.py` |

---

## Optional / Development

Not required to run `main.py`, but useful for development:

| Tool | Purpose |
|------|---------|
| `ipython` / `jupyter` | Interactive exploration of correlators and fit results |
| `pytest` | Unit tests (`tests/`); install via `requirements-dev.txt` |

---

## Version pinning

`requirements.txt` specifies **minimum versions** tested to work together. For fully reproducible environments, pin exact versions after verifying on your machine:

```bash
pip freeze > requirements-lock.txt
```

Example locked versions (update after local verification):

```
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.0
gvar==13.0.1
lsqfit==13.0.2
joblib==1.4.2
```
