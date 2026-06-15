# Lattice Scattering Analysis Pipeline

A modular Python pipeline for **lattice QCD spectroscopy and scattering analysis** of fully-charm tetraquark systems. It processes Monte Carlo correlation functions through GEVP diagonalization, Bayesian fitting, dispersion-relation calibration, and Lüscher scattering formalism — producing publication-ready figures and scattering observables.

**Current application:** the \(T_{cc\bar{c}\bar{c}}\) candidate **Tcccc6600**, with di-charmonium channels \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\), on \(N_f=2\) anisotropic ensembles at \(L=12,16\) and \(m_\pi\approx 420\) MeV.

---

## Why This Matters

Fully-charm tetraquarks \(T_{cc\bar{c}\bar{c}}\) are among the most striking exotic-hadron candidates seen at the LHC. Since 2020, LHCb, ATLAS, and CMS have reported multiple structures in the di-\(J/\psi\) spectrum — including **\(X(6600)\)**, **\(X(6900)\)**, and related states — whose internal structure and quantum numbers remain central open questions in QCD.

Lattice QCD provides a **first-principles** way to study these states: it computes the \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) interactions directly from the QCD Lagrangian, without model-dependent assumptions about tetraquark wave functions or meson-exchange potentials.

**Main physics results enabled by this pipeline:**

- **First lattice QCD evidence for a \(2^{++}\) resonance** in the \(J/\psi\,J/\psi\) sector near 6.6 GeV, with mass and width compatible with the broad structure **\(X(6600)\)** reported by ATLAS and CMS.
- **Consistency with experiment:** the preferred \(J^{PC}=2^{++}\) assignment aligns with the CMS angular analysis published in [Nature **648**, 58 (2025)](https://www.nature.com/articles/s41586-025-09278-2) ([arXiv:2506.07944](https://arxiv.org/abs/2506.07944) [hep-ex]).
- **Scalar–tensor decomposition:** separate \(0^{++}\) and \(2^{++}\) scattering amplitudes extracted via the Lüscher formalism, after GEVP removes operator mixing on the lattice.

This repository contains the **analysis code** and **pre-generated figures** behind those studies. Raw lattice correlators are not included (see [Data availability](#data-availability)).

### Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

### Experimental reference

- CMS Collaboration, *Observation of a resonance-like structure in the \(\mathrm{J}/\psi\)-pair invariant mass spectrum*, Nature **648**, 58 (2025). [arXiv:2506.07944](https://arxiv.org/abs/2506.07944) [hep-ex]

---

## Example Results

Representative outputs for **\(L=12\)** (`L12M420_EV170`). Full PDFs for \(L=12\) and \(L=16\) are in [`result/Tcccc6600/`](result/Tcccc6600/).

### GEVP (before / after diagonalization)

<p align="center">
  <img src="docs/figures/GEVP_before_L12.png" alt="GEVP matrix before diagonalization" width="48%" />
  <img src="docs/figures/GEVP_after_L12.png" alt="GEVP matrix after diagonalization" width="48%" />
</p>

### GEVP eigenvectors

<p align="center">
  <img src="docs/figures/GEVP_eigenvector_L12.png" alt="GEVP eigenvectors" width="55%" />
</p>

### Effective mass \(E_n\)

<p align="center">
  <img src="docs/figures/En_tetraquark_L12.png" alt="Tetraquark effective mass" width="48%" />
  <img src="docs/figures/En_meson_L12.png" alt="Meson effective mass" width="48%" />
</p>

### Overlap factors \(Z_n/Z_0\) and dispersion \(E_n^2\)

<p align="center">
  <img src="docs/figures/Zn_meson_L12.png" alt="Overlap factors Zn" width="48%" />
  <img src="docs/figures/Dispersion_meson_L12.png" alt="Dispersion relation" width="48%" />
</p>

### Scattering observables

<p align="center">
  <img src="docs/figures/K_s_scattering_L12.png" alt="Scattering K(s)" width="48%" />
  <img src="docs/figures/kcot_scattering_L12.png" alt="k cot delta_0" width="48%" />
</p>

---

## Pipeline Overview

The workflow is split into **two scripts** (see [docs/RUNNING.md](docs/RUNNING.md)):

```
data/<system>/raw/*.npy
        │
        ├─► run_resample.py  (jackknife / bootstrap)
        │         └─► data/<system>/resampled/*.npy
        │
        └─► main.py
              ├─► GEVP  →  effective-operator correlators
              ├─► Fit \(E_n\), \(Z_n\)  →  En / Zn plots
              ├─► Dispersion fit  →  \(\xi\)  →  Dispersion plot
              └─► Scattering (Lüscher zeta)  →  \(K_s\), \(k\cot\delta_0\) plots
```

| Step | Script | Output |
|------|--------|--------|
| Resampling | `run_resample.py` | Per-configuration energies and \(\xi\) in `data/<system>/resampled/` |
| Analysis + plots | `main.py` | PDF figures in `result/<system>/` |

**What `main.py` does (in order):**

1. Load raw correlators (and resampled data if scattering is enabled)
2. Build FVE matrix, solve GEVP, plot before/after matrices and eigenvectors
3. Fit effective masses; plot \(E_n\) and (for meson runs) \(Z_n/Z_0\)
4. Fit dispersion relation; plot \(E_n^2\) vs \(n^2\)
5. Run scattering analysis; plot \(K(s)\) and \(k\cot\delta_0\)

---

## Quick Start

### Requirements

- Python 3.10+
- TeX distribution (MacTeX / TeX Live) for LaTeX plot labels
- See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) and [docs/RUNNING.md](docs/RUNNING.md)

```bash
git clone https://github.com/Geng-Li-1995/lattice_scattering.git
cd lattice_scattering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run (from project root)

```bash
# Step 1 — resampling (required before scattering)
python run_resample.py

# Step 2 — analysis and plotting
python main.py
```

> Scattering needs resampled files under `data/Tcccc6600/resampled/`. Raw `.npy` correlators must be placed under `data/Tcccc6600/raw/` locally — they are **not** in this repository.

---

## Data availability

| Content | In repository? |
|---------|----------------|
| Analysis source code | Yes |
| Configuration (`input/Tcccc6600_input.py`) | Yes |
| Result PDFs (`result/Tcccc6600/`) | Yes |
| README preview PNGs (`docs/figures/`) | Yes |
| Raw correlators (`data/**/raw/*.npy`) | **No** (~80 MB) |
| Resampled energies (`data/**/resampled/*.npy`) | **No** |

Expected file naming:

| Path | Shape / content |
|------|-----------------|
| `correlation_meson_L{Ns}M{M}_EV{EV}.npy` | `[channel, momentum, time, sample]` |
| `correlation_tetraquark_L{Ns}M{M}_EV{EV}.npy` | `[ch_src, mom_src, ch_snk, mom_snk, time, sample]` |
| `resample_En_{type}_L{Ns}M{M}_EV{EV}.npy` | Jackknife/bootstrap energies |
| `resample_ksi_meson_L{Ns}M{M}_EV{EV}.npy` | Dispersion scale \(\xi\) |

Output PDF naming: `En_<type>_<tag>.pdf`, `GEVP_before_<tag>.pdf`, `K_s_scattering_<tag>.pdf`, etc., with `<tag>` = `L12M420_EV170`.

---

## Configuration

All physics parameters live in `input/Tcccc6600_input.py`.

**`InputControl`** — runtime switches (defaults shown):

```python
lattice_Ns: int = 12
is_meson_analysis: bool = True      # meson-only when True (tetraquark disabled)
is_tetraquark_analysis: bool = True  # set False when running meson analysis
is_gevp: bool = True
run_scattering: bool = True
resample_type: str = "jackknife"    # or "bootstrap"

# Scattering channel indices and K(s) fit momentum subsets
ch_meson_a: int = 1
ch_meson_b: int = 1
ch_tetra: int = 1
fit_mom_by_ns: Dict[int, List[int]] = {12: [0, 1, 2], 16: [0, 1]}
```

**Analysis mode:** `InputControl.__post_init__` enforces mutual exclusion — if `is_meson_analysis=True`, tetraquark analysis is turned off. For the **GEVP + tetraquark** pipeline, set `is_meson_analysis=False`, `is_tetraquark_analysis=True`. Run separately with `is_meson_analysis=True` for meson \(Z_n\) and dispersion plots.

**`ENSEMBLE_DB`** — per-ensemble channel lists, fit windows (`tmin_arry`), Bayesian priors, GEVP times, and \(a^{-1}\).

| \(L\) | \(N_t\) | \(m_\pi\) (MeV) | EV | \(a^{-1}\) (GeV) |
|-------|---------|-----------------|-----|------------------|
| 12 | 96 | 420 | 170 | 7.219 |
| 16 | 128 | 420 | 120 | 7.219 |

Scattering combines both volumes via `InputControl.Ns_list = [12, 16]`.

---

## Project Structure

```
lattice_scattering/
├── main.py                  # Analysis + plotting
├── run_resample.py          # Jackknife / bootstrap resampling
├── input/
│   ├── config.py            # BuildConfig → Config
│   ├── Tcccc6600_input.py   # Physics parameters
│   ├── selector.py          # Correlator / model selection
│   └── types.py             # Type aliases and naming conventions
├── data/io.py               # Load raw and resampled .npy files
├── analysis/
│   ├── gevp.py              # GEVP solver
│   ├── fitting.py           # Effective-mass and dispersion fits
│   ├── scattering.py        # Lüscher scattering analysis
│   ├── models.py            # Cosh fit models and priors
│   └── utils.py             # Shared fit lookup helpers
├── statistics/              # Jackknife, bootstrap, resample driver
├── plotting/
│   ├── plot_set.py          # Shared colors, fonts, figure sizes
│   ├── plot_gevp.py
│   ├── plot_mass.py
│   └── plot_scattering.py
├── docs/figures/            # PNG previews for README
└── result/Tcccc6600/        # Generated PDF figures
```

---

## Methods (brief)

| Method | Role in this project |
|--------|----------------------|
| **GEVP** | Diagonalize the \(\eta_c\eta_c\)–\(J/\psi\,J/\psi\) correlation matrix to isolate effective operators and suppress contaminations |
| **Multi-state cosh fit** | Extract energies \(E_n\) and overlaps \(Z_n\) with Bayesian priors (`lsqfit` + `gvar`) |
| **Dispersion relation** | Calibrate lattice spacing \(\xi\) from \(E_n^2\) vs \(n^2\) |
| **Lüscher zeta function** | Convert finite-volume energies to \(k\cot\delta_0\) and \(K(s)=\sqrt{s}/(k\cot\delta)\) |
| **Jackknife / bootstrap** | Statistical errors on per-configuration fits |

---

## Technical Highlights

- **Scientific Python:** `numpy`, `scipy.linalg`, `matplotlib`, `joblib`
- **Bayesian fitting:** `gvar` / `lsqfit` with full error propagation on fit curves
- **Modular design:** configuration, I/O, analysis, statistics, and plotting are decoupled
- **Extensible:** new systems via `input/<system>_input.py` + system name in `main.py`

---

## Notes

- Plotting calls `plt.show()`; on headless systems use matplotlib backend `Agg` (see [docs/RUNNING.md](docs/RUNNING.md)).
- The zeta lookup table is cached at `data/zeta/zeta_00_rest_array.npy` after first generation.
- `plot_meff` / `plot_dispersion` flags in `InputControl` are reserved; `main.py` currently runs all applicable plots for the active analysis mode.

---

## License

Not specified. Contact the maintainer before redistribution.
