# Lattice Scattering Analysis Pipeline

A modular Python pipeline for **lattice QCD spectroscopy and scattering analysis**, built to extract hadronic energies and scattering observables from Monte Carlo correlation functions.

The current application targets the fully-charm tetraquark system **Tcccc6600**, with analysis channels \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\), across multiple lattice volumes (\(L = 12, 16\)).

---

## Project Goal

In lattice QCD, extracting reliable hadron masses and scattering information from noisy correlation functions requires a chain of specialized numerical methods: operator mixing removal, Bayesian fitting, dispersion-relation calibration, and finite-volume scattering formalism.

This project implements that full analysis workflow as a **reproducible, configurable pipeline** — from raw `.npy` correlators to publication-ready PDF plots and scattering amplitudes.

**Scientific objectives:**

- Diagonalize multi-channel tetraquark correlators via the **Generalized Eigenvalue Problem (GEVP)** to isolate effective operators
- Extract ground-state and excited-state energies \(E_n\) and overlap factors \(Z_n\) through **multi-state cosh fits**
- Determine the lattice spacing scale \(\xi\) from **dispersion relations**
- Compute scattering observables \(K_s\) and \(k\cot\delta_0\) using the **Lüscher formalism** with precomputed zeta functions
- Quantify statistical uncertainties via **jackknife** and **bootstrap** resampling

---

## Key Features

| Module | Description |
|--------|-------------|
| **GEVP solver** | Generalized eigenvalue decomposition \(C(t_1)v = \lambda\, C(t_0)v\) with eigenvector ordering |
| **Bayesian fitting** | Two- and three-state cosh models using `lsqfit` + `gvar` with configurable priors |
| **Dispersion analysis** | Linear fit of \(E_n^2\) vs. lattice momentum \(n^2\) to extract \(\xi\) |
| **Scattering analysis** | Rest-frame Lüscher zeta function lookup; \(K_s = \sqrt{s}\,/\,(k\cot\delta)\) extraction |
| **Resampling statistics** | Jackknife (default) and bootstrap error estimation on per-configuration fits |
| **Automated plotting** | LaTeX-rendered PDF outputs for GEVP matrices, effective masses, dispersion, and scattering |

---

## Technical Highlights

Skills and practices demonstrated in this project:

- **Scientific Python**: `numpy`, `scipy.linalg`, `matplotlib`, `joblib` for numerics, linear algebra, visualization, and parallel precomputation
- **Bayesian inference**: Nonlinear least-squares fitting with error propagation via `gvar` / `lsqfit` — standard tools in lattice QCD data analysis
- **Modular architecture**: Separation of configuration, I/O, analysis, statistics, and plotting layers; new physics systems added by dropping in an `input/<system>_input.py` file
- **Dataclass-driven configuration**: `InputControl` + `ENSEMBLE_DB` pattern for ensemble-specific physics parameters (channels, fit windows, priors, GEVP times)
- **Reproducible workflow**: Single-entry `main.py` orchestrates the full pipeline; intermediate and final results saved to structured directories
- **Performance awareness**: Zeta function lookup table precomputed once and cached; `joblib` parallelization for expensive summations

---

## Analysis Workflow

```
Raw correlators (.npy)
        │
        ├─► GEVP diagonalization ──► Effective-operator correlators
        │                                    │
        │                                    ├─► Effective-mass fit (cosh model)
        │                                    │       ├─► En, Zn plots
        │                                    │       └─► Dispersion relation → ξ
        │                                    │
        └─► Resampled energies + ξ ──► Scattering analysis (Lüscher zeta)
                                             ├─► Ks plot
                                             └─► k·cot(δ) plot
```

---

## Project Structure

```
lattice_scattering/
├── main.py                  # Pipeline entry point
├── input/                   # Configuration layer
│   ├── config.py            # BuildConfig → Config dataclass
│   ├── Tcccc6600_input.py   # Physics parameters for Tcccc6600
│   └── selector.py          # Data / model selection
├── data/
│   ├── io.py                # Data loading
│   ├── <system>/raw/        # Raw correlation functions
│   ├── <system>/resampled/  # Jackknife/bootstrap outputs
│   └── zeta/                # Precomputed Lüscher zeta lookup table
├── analysis/
│   ├── gevp.py              # GEVP solver
│   ├── fitting.py           # Effective-mass & dispersion fits
│   ├── scattering.py        # Scattering amplitude extraction
│   └── models.py            # Fit models and Bayesian priors
├── statistics/
│   ├── jackknife.py
│   ├── bootstrap.py
│   └── resample.py
├── plotting/
│   ├── plot_gevp.py
│   ├── plot_mass.py
│   └── plot_scattering.py
└── result/<system>/         # Output PDF figures
```

---

## Quick Start

### Requirements

- **Python 3.10+**
- See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for full dependency details
- See [docs/RUNNING.md](docs/RUNNING.md) for the complete running guide

| Package | Role |
|---------|------|
| `numpy` | Array operations |
| `scipy` | Generalized eigenvalue solver |
| `matplotlib` | Plotting (LaTeX rendering enabled) |
| `gvar` | Error propagation |
| `lsqfit` | Bayesian nonlinear fitting |
| `joblib` | Parallel zeta precomputation |

A TeX distribution (MacTeX, TeX Live, etc.) is required for LaTeX-rendered plot labels.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

Run from the **project root**:

```bash
python main.py
```

The default pipeline (`main.py`):

1. Build configuration from `input/Tcccc6600_input.py`
2. Load correlators from `data/Tcccc6600/raw/`
3. Apply GEVP to tetraquark data and plot correlation matrices
4. Fit effective masses \(E_n\) and weights \(Z_n\)
5. Fit dispersion relations and plot \(\xi\)
6. Run scattering analysis and plot \(K_s\), \(k\cot\delta\)
7. *(Optional)* Run jackknife/bootstrap resampling (`run_resample=True`)

> **Note:** Scattering analysis requires resampled files in `data/<system>/resampled/`. Generate them first by setting `run_resample=True` — see [docs/RUNNING.md §5](docs/RUNNING.md#5-running-the-pipeline).

---

## Configuration

Analysis parameters live in `input/<system>_input.py`:

**`InputControl`** — runtime switches and lattice setup:

```python
@dataclass
class InputControl:
    lattice_Ns: int = 12
    is_tetraquark_analysis: bool = True
    is_gevp: bool = True
    run_scattering: bool = True
    run_resample: bool = False
    resample_type: str = "jackknife"  # or "bootstrap"
```

**`ENSEMBLE_DB`** — per-ensemble physics: channel names, momentum grids, fit time windows, Bayesian priors, GEVP reference times, and inverse lattice spacing \(a^{-1}\).

To analyze a new system, copy `Tcccc6600_input.py`, fill in the database, and update the system name in `main.py`.

---

## Data Layout

### Raw correlators (`data/<system>/raw/`)

| File | Shape |
|------|-------|
| `correlation_meson_L{Ns}M{M}_EV{EV}.npy` | `[channel, momentum, time, sample]` |
| `correlation_tetraquark_L{Ns}M{M}_EV{EV}.npy` | `[ch_src, mom_src, ch_snk, mom_snk, time, sample]` |

### Resampled outputs (`data/<system>/resampled/`)

Generated by `run_resample_statistics`; required for scattering analysis:

| File | Content |
|------|---------|
| `resample_En_meson_L{Ns}M{M}_EV{EV}.npy` | Per-channel/momentum resampled energies |
| `resample_En_tetraquark_L{Ns}M{M}_EV{EV}.npy` | Tetraquark resampled energies |
| `resample_ksi_meson_L{Ns}M{M}_EV{EV}.npy` | Dispersion-relation scale \(\xi\) |

---

## Output Figures

PDFs are saved to `result/<system>/`. Example outputs (included in this repo):

| Figure | Filename pattern |
|--------|-----------------|
| GEVP matrix (before / after) | `GEVP_before_<tag>.pdf`, `GEVP_after_<tag>.pdf` |
| Eigenvectors | `GEVP_eigenvector_<tag>.pdf` |
| Effective mass | `En_<type>_<tag>.pdf` |
| Overlap factors | `Zn_<type>_<tag>.pdf` |
| Dispersion relation | `Dispersion_meson_<tag>.pdf` |
| Scattering \(K_s\) | `K_s_scattering_<tag>.pdf` |
| Scattering \(k\cot\delta\) | `kcot_scattering_<tag>.pdf` |

`<tag>` follows the convention `L12M420_EV170`; `<type>` is `meson` or `tetraquark`.

---

## Current Ensembles

| \(L\) | \(N_t\) | Pion mass (MeV) | Eigenvectors | \(a^{-1}\) (GeV) |
|-------|---------|-----------------|--------------|------------------|
| 12 | 96 | 420 | 170 | 7.219 |
| 16 | 128 | 420 | 120 | 7.219 |

Scattering analysis combines results from both volumes (`InputControl.Ns_list = [12, 16]`).

---

## Notes

- Scattering analysis requires pre-generated resampled files in `resampled/`. Set `run_resample=True` first to produce them.
- Plotting modules call `plt.show()`; on headless systems, set the matplotlib backend to `Agg`.
- Some channel indices and momentum selections in `analysis/scattering.py` are currently hardcoded for Tcccc6600.

---

## Version Control

This project uses Git. Large binary files (`.npy` correlators, PDF outputs) are excluded via `.gitignore`; only source code and documentation are tracked.

```bash
# First-time setup (already done if .git/ exists)
git init

# Stage and commit
git add .
git commit -m "Initial commit: lattice QCD scattering analysis pipeline"

# Optional: push to GitHub
git remote add origin https://github.com/<user>/lattice_scattering.git
git branch -M main
git push -u origin main
```

Place local data files under `data/<system>/raw/` after cloning — they are not stored in the repository (~80 MB). See [docs/RUNNING.md](docs/RUNNING.md) for the expected file layout.

To **track data or results** in Git (e.g. for a private repo), remove the corresponding lines from `.gitignore`.

---

## Background

This pipeline applies standard lattice QCD spectroscopy techniques:

- **GEVP** (Lüscher & Weisz, 1990s): removes contaminating states from variational operator bases
- **Effective-mass fitting**: multi-exponential cosh/sinh models for energy extraction
- **Lüscher formula** (2000s): relates finite-volume energy shifts to scattering phase shifts via the zeta function

Relevant domains: **computational physics**, **scientific computing**, **Bayesian data analysis**, **HPC / lattice gauge theory**.

---

## License

Not specified. Contact the maintainer before redistribution.
