# Lattice QCD Tetraquark Scattering

[![CI](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml)

End-to-end lattice QCD analysis for tetraquark spectroscopy and finite-volume scattering. The pipeline ingests Monte Carlo correlation functions up to **six dimensions**, runs generalized eigenvalue decomposition and Bayesian multi-state fits on **401** jackknife replicas, and extracts Lüscher scattering observables with correlated uncertainties carried through every stage.

| | |
|--|--|
| **Systems** | `Tcccc6600`, `X3872`, `Zc3900` — `BuildConfig("<System>")` in `main.py` |
| **Reference** | Tcccc6600: \(\eta_c\eta_c\) / \(J/\psi\,J/\psi\), \(L=12,16\), \(a_t^{-1}=7.219\) GeV |
| **Data** | Raw / resampled `.npy` local only (`data/<system>/`); not in git |

---

## Technical capabilities

### Scale

| Quantity | Typical value (Tcccc6600) |
|----------|---------------------------|
| Raw tetraquark array rank | **6D** `[ch_src, mom_src, ch_snk, mom_snk, time, sample]` |
| Elements per raw file | \(\sim10^6\)–\(10^7\) `float64` (~30–40 MiB / volume) |
| Gauge ensemble (`sample` axis) | **401** jackknife replicas |
| Time extent \(N_t\) | 96 (\(L=12\)) / 128 (\(L=16\)) |
| Jackknife resampling cost | **401×** full GEVP + fit pass per volume when `run_resample=True` |
| Lüscher zeta grid | \(10^5\) \(q^2\) points, built once in parallel (`joblib`) |

Meson correlators are **4D**; tetraquark raw data is the dominant memory and tensor cost. After GEVP both branches are 4D for fitting.

### Methods & implementation

| Area | What the code does |
|------|-------------------|
| **I/O & types** | Shape-safe `Correlator4D`, `TetraquarkCorrelator`, `AnalysisCorrelators`; raw and resampled `.npy` loaders with ensemble tags (`L{Ns}M{M}_EV{EV}`) |
| **Configuration** | `input/input_<System>.py` + `ENSEMBLE_DB`: channels, \(n^2\) lists, GEVP times, fit windows, priors — no hard-coded physics in analysis modules |
| **GEVP** | Assemble full finite-volume matrix from 6D tetraquark data; solve \(C(t_1)\psi = \lambda C(t_0)\psi\) (`scipy.linalg.eig`); rotate correlators with `numpy.einsum`; extract diagonal GEVP levels |
| **Spectroscopy fits** | Registry-based cosh models (`models.py`); `lsqfit` nonlinear fit with `gvar` priors; meson 2-state / tetraquark 3-state \(E_n\), \(Z_n\); dispersion \(E_n^2(n^2)\) for scale \(\xi\) |
| **\(t_{\min}\) & ratio** | Symmetric window scan; 4Q/2Q ratio series and fits (`fit_tmin.py`); combined stability plots (`plot_tmin.py`) |
| **Resampling** | Leave-one-out jackknife (`statistics/jackknife.py`) or bootstrap; write per-sample `En` and \(\xi\) to `resampled/` for downstream scattering |
| **Scattering** | Rest-frame and moving-frame Lüscher zeta; per-jackknife-sample \(K(s)\) and \(k\cot\delta_0\); linear / quadratic phase-shift fits; cross-volume combination (`Ns_list`) |
| **Plotting** | Shared style (`plot_set.py`); GEVP matrices, \(E_n\), \(t_{\min}\), scattering figures; LaTeX rendering; PNG or PDF via `plot_format` |
| **Testing & CI** | Unit tests on synthetic arrays (no multi‑MiB lattice files); GitHub Actions on Python **3.10** and **3.12** |

### Software design

- **Modular stages:** `data/` → `analysis/` → `statistics/` → `plotting/` — each stage switch-controlled from `InputControl`.
- **One entrypoint:** `main.py` orchestrates meson and tetraquark branches independently; scattering can rerun from cached `resampled/` without raw correlators.
- **Extensible systems:** new tetraquark setup = one `input_<System>.py` + local `data/<System>/raw/` + `BuildConfig` line — analysis code unchanged.
- **Reproducible artefacts:** zeta tables under `data/zeta/`; jackknife energies under `data/<system>/resampled/`; figures under `result/<system>/`.

**Stack:** Python 3.10+ · NumPy · SciPy · gvar · lsqfit · joblib · Matplotlib · pytest

---

## Physics (Tcccc6600)

Fully-charm tetraquarks \(T_{cc\bar{c}\bar{c}}\) are exotic-hadron candidates at the LHC. This analysis chain targets \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) scattering and comparison with LHC data.

**Key result:** a **\(2^{++}\)** candidate in \(J/\psi\,J/\psi\) near **6.6 GeV**, compatible with **\(X(6600)\)** ([Nature **648**, 58 (2025)](https://www.nature.com/articles/s41586-025-09278-2); [arXiv:2506.07944](https://arxiv.org/abs/2506.07944)), with separate **\(0^{++}\)** and **\(2^{++}\)** amplitudes from one GEVP spectrum.

**Scattering observables:** \(K(s)\) vs \(s=m_{\rm CM}^2\) encodes S-wave interaction strength; \(k\cot\delta_0\) gives the phase shift \(\delta_0\). **Zeros of \(K(s)\) are S-matrix poles** — resonance or bound-state positions to compare with experimental enhancements such as \(X(6600)\) in double-\(J/\psi\) production.

---

## Example results

\(L=12\) (`L12M420_EV170`) unless noted; \(t_{\min}\) scan: \(L=16\).

### Generalized Eigenvalue Problem (GEVP)

<p align="center">
  <img src="result/Tcccc6600/GEVP_before_L12M420_EV170.png" alt="GEVP matrix before diagonalization" width="48%" />
  <img src="result/Tcccc6600/GEVP_after_L12M420_EV170.png" alt="GEVP matrix after diagonalization" width="48%" />
</p>

### Effective mass \(E_n\) (meson ← · tetraquark →)

<p align="center">
  <img src="result/Tcccc6600/En_meson_L12M420_EV170.png" alt="Meson En" width="48%" />
  <img src="result/Tcccc6600/En_tetraquark_L12M420_EV170.png" alt="Tetraquark En" width="48%" />
</p>

### \(t_{\min}\) scan — `E2_mom0` & `E2_mom1` (\(J/\psi\,J/\psi\), \(L=16\))

3-state ○, ratio ×, Combine band (GeV, ±10× jackknife error). Left: \(n^2=0\); right: \(n^2=1\).

<p align="center">
  <img src="result/Tcccc6600/E2_mom0_tmin_ratio_L16M420_EV120.png" alt="E2_mom0 tmin ratio" width="48%" />
  <img src="result/Tcccc6600/E2_mom1_tmin_ratio_L16M420_EV120.png" alt="E2_mom1 tmin ratio" width="48%" />
</p>

<p align="center"><sub><code>E2_mom0_tmin_ratio_L16M420_EV120</code> · <code>E2_mom1_tmin_ratio_L16M420_EV120</code></sub></p>

### Scattering — \(L=12+16\)

<p align="center">
  <img src="result/Tcccc6600/K_s_scattering.png" alt="K(s)" width="48%" />
  <img src="result/Tcccc6600/kcot_scattering.png" alt="k cot delta" width="48%" />
</p>

---

## Pipeline

```
data/<system>/raw/*.npy  […, sample=401]
        │
        ▼  main.py  (input/input_<System>.py switches)
   run_resample_statistics()     →  resampled/*.npy     (401× jackknife if enabled)
   process_GEVP()                →  4D tetraquark correlators
   effective_mass()              →  En, Zn, dispersion
   plot_tmin_workflow()          →  optional t_min + ratio plots
   run_scattering_analysis()     →  K(s), k cot δ₀
        │
        ▼
   result/<system>/*.{png,pdf}
```

**Execution order:** build `Config` → resample (optional) → meson branch → tetraquark branch (GEVP, fits, t_min) → scattering. Meson/tetraquark switches are independent.

---

## Code layout

```
lattice_scattering/
├── main.py                         # entry: meson/tetraquark branches → scattering → figures
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .github/
│   └── workflows/
│       └── ci.yml                  # pytest on Python 3.10 & 3.12
│
├── input/
│   ├── __init__.py
│   ├── config.py                   # BuildConfig, Config, ENSEMBLE_DB, ensemble tags
│   ├── input_Tcccc6600.py          # η_cη_c / J/ψJ/ψ, L=12+16
│   ├── input_X3872.py              # πJ/ψ, ρη_c, DD*, D*D*
│   └── input_Zc3900.py
│
├── data/
│   ├── correlators.py              # Correlator4D, TetraquarkCorrelator, AnalysisCorrelators
│   ├── io.py                       # read/write raw & resampled .npy
│   ├── <System>/                   # Tcccc6600 | X3872 | Zc3900 (not in git)
│   │   ├── raw/
│   │   │   ├── correlation_meson_L{Ns}M{M}_EV{EV}.npy
│   │   │   └── correlation_tetraquark_L{Ns}M{M}_EV{EV}.npy
│   │   └── resampled/
│   │       ├── resample_En_{meson,tetraquark}_*.npy
│   │       └── resample_ksi_meson_*.npy
│   └── zeta/                       # precomputed Lüscher ζ(q²); moving-frame refs per system
│
├── analysis/
│   ├── __init__.py
│   ├── gevp.py                     # process_GEVP, solve_GEVP, eigenvector sorting
│   ├── fit_mass.py                 # RunFitting: effective mass, dispersion, Z_n
│   ├── fit_tmin.py                 # t_min scan (cosh & ratio), ratio_scan_lookup
│   ├── scattering.py               # Lüscher K(s), k cot δ, rest & moving frames
│   └── models.py                   # MathModels, MODEL_REGISTRY (cosh, dispersion, …)
│
├── statistics/
│   ├── __init__.py
│   ├── jackknife.py                # Jackknife resampler (401 replicas)
│   ├── bootstrap.py                # Bootstrap resampler (optional)
│   └── resample.py                 # run_resample_statistics → data/<system>/resampled/
│
├── plotting/
│   ├── __init__.py
│   ├── plot_set.py                 # BasePlotter, matplotlib style, save_figure
│   ├── plot_gevp.py                # GEVP before/after matrices & eigenvectors
│   ├── plot_mass.py                # E_n, Z_n, dispersion; delegates t_min to TminPlotter
│   ├── plot_tmin.py                # TminPlotter: cosh & ratio t_min stability
│   └── plot_scattering.py          # K(s), k cot δ vs. (k/m_π)²
│
├── result/
│   └── <System>/                   # output figures (.png / .pdf); tracked as examples
│
├── docs/
│   ├── RUNNING.md                  # install, data layout, control flags
│   ├── TESTING.md                  # pytest scope & fixtures
│   └── DEPENDENCIES.md             # Python packages & optional LaTeX
│
└── tests/
    ├── conftest.py                 # shared fixtures (synthetic correlators)
    ├── test_jackknife.py
    ├── test_io.py
    ├── test_fit_mass.py
    ├── test_fit_tmin.py
    ├── test_scattering.py
    ├── test_scattering_fit.py
    ├── test_plot_set.py
    └── test_plot_tmin.py
```

New system: copy `input/input_<System>.py`, add `data/<System>/raw/`, change `BuildConfig(...)` in `main.py`.

---

## Data

All arrays: **`float64`**. **`sample` = 401** jackknife replicas. **`mom`** = \(n^2\) array index. **`EV`** in filenames = distillation eigenvectors (not `sample` length).

### Shapes

| Object | Axes |
|--------|------|
| Meson raw | `[channel, mom, time, sample]` |
| Tetraquark raw (6D) | `[ch_src, mom_src, ch_snk, mom_snk, time, sample]` |
| After GEVP | `[channel, mom, time, sample]` |
| Resampled `En` | `[channel, mom, sample]` |

### Raw files (Tcccc6600)

| File | Shape | Size |
|------|-------|------|
| `correlation_meson_L12M420_EV170.npy` | `[2, 10, 96, 401]` | 5.9 MiB |
| `correlation_tetraquark_L12M420_EV170.npy` | `[2, 5, 2, 5, 96, 401]` | 29 MiB |
| `correlation_meson_L16M420_EV120.npy` | `[2, 10, 128, 401]` | 7.9 MiB |
| `correlation_tetraquark_L16M420_EV120.npy` | `[2, 5, 2, 5, 128, 401]` | 39 MiB |

~**82 MiB** raw per system (\(L=12+16\)). X3872/Zc3900 (\(L=16\)): meson `[6,5,128,401]`, tetraquark `[4,2,4,2,128,401]`.

### Resampled (`data/<system>/resampled/`)

| Pattern | Shape (example) |
|---------|-----------------|
| `resample_En_meson_*` | `[6, 5, 401]` |
| `resample_En_tetraquark_*` | `[1, 3, 401]` |
| `resample_ksi_meson_*` | `[6, 401]` |

Scattering uses **401-sample** correlated energies (no raw reload). `run_resample=True` costs **401×** GEVP+fit per volume; scattering/plotting afterward is cheap.

### Systems & ensembles

| System | \(L\) | EV | Channels |
|--------|-------|-----|----------|
| Tcccc6600 | 12, 16 | 170 / 120 | \(\eta_c\eta_c\), \(J/\psi J/\psi\) |
| X3872, Zc3900 | 16 | 70 | \(\pi J/\psi\), \(\rho\eta_c\), \(DD^*\), \(D^*D^*\) |

Tcccc6600: \(N_t=96/128\), \(m_\pi=420\) MeV, \(a_t^{-1}=7.219\) GeV; scattering uses `Ns_list = [12, 16]`.

---

## Usage

```bash
git clone https://github.com/Geng-Li-1995/lattice_scattering.git
cd lattice_scattering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
MPLBACKEND=Agg pytest && python main.py
```

Details: [docs/RUNNING.md](docs/RUNNING.md) · [docs/TESTING.md](docs/TESTING.md) · [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)

### Switches (`input/input_<System>.py`)

```python
lattice_Ns = 12
is_meson_analysis = True
is_tetraquark_analysis = True
is_gevp = True
run_tmin = True
is_ratio = True
run_resample = False   # True → 401× jackknife
run_scattering = True
plot_format = "png"    # or "pdf"
resample_type = "jackknife"
```

### Figure naming (`plot_format` extension; tag `L{Ns}M{M}_EV{EV}`)

| Plot | Pattern |
|------|---------|
| GEVP | `GEVP_{before,after,eigenvector}_{tag}` |
| \(E_n\) | `En_{meson,tetraquark}_{tag}` |
| \(t_{\min}\) | `E{n}_mom{n2}_tmin_{tag}` or `*_tmin_ratio_*` |
| Scattering | `K_s_scattering`, `kcot_scattering` (no volume tag) |

`E{n}` = tetraquark channel index; `mom{k}` = \(n^2=k\).

---

## Tests & CI

`pytest` on Python 3.10 & 3.12 without lattice data ([CI workflow](.github/workflows/ci.yml)): jackknife, I/O, scattering algebra, fit/t_min lookups.

---

## Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

---

## License

Not specified. Contact the maintainer before redistribution.
