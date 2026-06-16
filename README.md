# Lattice QCD Scattering Analysis Pipeline

**End-to-end Python pipeline for lattice QCD spectroscopy and finite-volume scattering analysis** — ingests large Monte Carlo correlation-function datasets, performs generalized eigenvalue decomposition (GEVP), Bayesian multi-state fits, and Lüscher scattering extraction with full statistical error propagation.

**Showcase system:** fully-charm tetraquark **Tcccc6600** (\(\eta_c\eta_c\), \(J/\psi\,J/\psi\)) on \(N_f=2\) anisotropic ensembles (\(L=12,16\), \(N_t=96/128\), \(m_\pi\approx 420\) MeV, **400 gauge configurations** per volume, **120–170 distillation eigenvectors** depending on \(L\)).

---

## Highlights

| Domain | What this repo demonstrates |
|--------|----------------------------|
| **HPC / large-scale numerics** | Batch processing of high-dimensional correlator arrays (4D meson, 6D tetraquark); jackknife over **400 gauge configs**, each rerunning GEVP + fits |
| **Scientific computing** | Generalized eigenvalue problems (`scipy.linalg.eig`), tensor contractions (`numpy.einsum`), Lüscher zeta summation on \(10^5\)-point grids |
| **Statistical inference** | Bayesian nonlinear fits (`lsqfit` + `gvar`); jackknife / bootstrap resampling with correlated uncertainties end-to-end |
| **Parallel computing** | `joblib` parallel precomputation of zeta lookup tables (`n_jobs=-1`) |
| **Software engineering** | Modular pipeline (I/O → analysis → statistics → plotting); typed dataclass wrappers; config-driven `ENSEMBLE_DB`; publication-ready figures (`plot_format`: PNG or PDF) |

**Stack:** Python 3.10+ · NumPy · SciPy · gvar · lsqfit · joblib · Matplotlib (LaTeX)

---

## Scientific Context

Fully-charm tetraquarks \(T_{cc\bar{c}\bar{c}}\) are among the most striking exotic-hadron candidates seen at the LHC. Lattice QCD provides a **first-principles, model-independent** route to \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) interactions and scattering amplitudes.

**Key results enabled by this pipeline:**

- First lattice QCD evidence for a **\(2^{++}\) resonance** in the \(J/\psi\,J/\psi\) sector near 6.6 GeV, compatible with the broad **\(X(6600)\)** structure reported by ATLAS and CMS.
- Preferred \(J^{PC}=2^{++}\) assignment consistent with the CMS angular analysis ([Nature **648**, 58 (2025)](https://www.nature.com/articles/s41586-025-09278-2); [arXiv:2506.07944](https://arxiv.org/abs/2506.07944)).
- Separate \(0^{++}\) and \(2^{++}\) scattering amplitudes via the Lüscher formalism after GEVP removes operator mixing.

| Method | Role |
|--------|------|
| **GEVP** | Diagonalize the \(\eta_c\eta_c\)–\(J/\psi\,J/\psi\) matrix; suppress contaminations |
| **Multi-state cosh fit** | Extract \(E_n\), \(Z_n\) with Bayesian priors |
| **Dispersion relation** | Calibrate lattice spacing \(\xi\) from \(E_n^2\) vs \(n^2\) |
| **Lüscher zeta function** | Finite-volume energies \(\to\) \(k\cot\delta_0\), \(K(s)\) |
| **Jackknife / bootstrap** | Per-configuration statistical errors on fits and scattering observables |

---

## Pipeline Architecture

Two-stage batch workflow — run from the **project root**:

```
Monte Carlo correlators          data/<system>/raw/*.npy
  [ch, mom, t, sample]           [ch_src, mom_src, ch_snk, mom_snk, t, sample]
         │
         ├─► run_resample.py          Step 1 — resampling (before scattering)
         │     jackknife / bootstrap over gauge configs
         │     full GEVP + fit per leave-one-out sample
         │         └─► data/<system>/resampled/*.npy
         │
         └─► main.py                    Step 2 — analysis + plots
               ├─► process_GEVP()       FVE matrix build + generalized eig (tetraquark)
               ├─► effective_mass()     Bayesian cosh fits (always)
               ├─► dispersion()         meson calibration (optional)
               └─► run_scattering_analysis()   Lüscher zeta → K(s), k cot δ₀
                         └─► result/<system>/*.{png|pdf}   (plot_format)
```

**Compute profile:** Step 1 is the heavy stage — \(O(N_{\mathrm{cfg}})\) full analysis passes (**400 jackknife leaves** per ensemble). Step 2 loads precomputed resampled energies and runs scattering fits plus figure generation. Zeta tables are built once and cached at `data/zeta/zeta_00_rest_array.npy`.

---

## Capabilities & Modules

| Capability | Module | Output |
|------------|--------|--------|
| Load raw / resampled `.npy` correlators | `data/io.py` | `RawCorrelators`, `resampled` |
| Typed array wrappers (shape-safe I/O) | `data/correlators.py` | `Correlator4D`, `TetraquarkCorrelator` |
| Build FVE matrix, solve GEVP | `analysis/gevp.py` | `AnalysisCorrelators` |
| Multi-state cosh fits (\(E_n\), \(Z_n\)) | `analysis/fitting.py` + `models.py` | `en_fit_list` |
| Dispersion calibration (\(\xi\)) | `analysis/fitting.py` | `disp_fit_list` (meson mode) |
| Jackknife / bootstrap resampling | `statistics/` | `data/<system>/resampled/*.npy` |
| Lüscher zeta scattering | `analysis/scattering.py` | `scattering_dict` → \(K(s)\), \(k\cot\delta_0\) |
| Unified plot styling | `plotting/plot_set.py` | TeX fonts, z-order, `save_figure()` |
| Figure output | `plotting/plot_*.py` | `result/<system>/*.{png,pdf}` (see naming below) |

**Design:** configuration, I/O, analysis, statistics, and plotting are decoupled. A new physics system needs only `input/<System>_input.py` and a one-line change in `main.py`.

---

## Code Structure

```
lattice_scattering/
├── main.py                  # GEVP → fits → plots → scattering
├── run_resample.py          # Jackknife / bootstrap → resampled energies
├── input/
│   ├── config.py            # BuildConfig → immutable Config dataclass
│   ├── Tcccc6600_input.py   # InputControl switches + ENSEMBLE_DB priors
│   ├── selector.py          # Correlator4D and fit model selection
│   └── types.py             # Type aliases
├── data/
│   ├── correlators.py       # Correlator4D, TetraquarkCorrelator, Raw/AnalysisCorrelators
│   └── io.py                # read_raw_files(), read_file()
├── analysis/
│   ├── gevp.py              # FVE matrix, scipy generalized eig, einsum rotation
│   ├── fitting.py           # RunFitting: effective_mass(), dispersion()
│   ├── scattering.py        # Lüscher zeta (joblib), K(s) / kcot fits
│   ├── models.py            # Cosh models, priors, MODEL_REGISTRY
│   └── utils.py             # en_fit_lookup, disp_fit_lookup, fve_offsets
├── statistics/
│   ├── jackknife.py / bootstrap.py
│   └── resample.py          # run_resample_statistics()
└── plotting/
    ├── plot_set.py          # RC_PARAMS, COLORS, save_figure()
    ├── plot_gevp.py
    ├── plot_mass.py         # En, Zn, Dispersion
    └── plot_scattering.py   # K_s, kcot
```

**Data types** (`data/correlators.py`):

| Variable | Type / shape |
|----------|----------------|
| `raw` | `RawCorrelators` — meson `[ch, mom, t, sample]`, tetraquark `[ch_src, mom_src, ch_snk, mom_snk, t, sample]`; `sample` = 400 gauge configs |
| `corr` | `AnalysisCorrelators` — both branches 4D after GEVP |
| `resampled` | Per-configuration `En` and meson \(\xi\) from jackknife/bootstrap |
| `en_fit_list` / `disp_fit_list` | Effective-mass / dispersion fit results |
| `scattering_dict` | Scattering observables per ensemble |

`mom` is the momentum quantum number \(n^2\) used as an **array index** (not a sequential 0…N−1 label). Channel/momentum lists come from `chan_momt_list` in `ENSEMBLE_DB`.

### Output figure naming

Extension comes from `plot_format` in `InputControl` (default **`png`**). Below, `{ext}` = `png` or `pdf`.

| Plot | Filename pattern | Scope |
|------|------------------|-------|
| GEVP matrix / eigenvector | `GEVP_{before,after,eigenvector}_L{Ns}M{M}_EV{EV}.{ext}` | Single ensemble (`lattice_Ns`) |
| Effective mass | `En_{meson,tetraquark}_L{Ns}M{M}_EV{EV}.{ext}` | Single ensemble |
| Overlap / dispersion | `Zn_meson_*.{ext}`, `Dispersion_meson_*.{ext}` | Single ensemble (meson mode) |
| Scattering | `K_s_scattering.{ext}`, `kcot_scattering.{ext}` | **Cross-ensemble** (`Ns_list`; no `L/M/EV` tag) |

### `main.py` execution order

1. `BuildConfig("Tcccc6600").build_config_from_control()`
2. `read_file()` — raw correlators; resampled files if `run_scattering=True`
3. `process_GEVP()` + GEVP plots — tetraquark mode only, if `is_gevp=True`
4. `effective_mass()` — **always runs** (stdout diagnostics)
5. `En_*` plot — if `plot_meff=True`; `Zn_meson_*` only if `is_meson_analysis=True`
6. Dispersion fit + plot — if `plot_dispersion=True` and meson mode
7. Scattering + `K_s_scattering.*`, `kcot_scattering.*` — if `run_scattering=True` and tetraquark mode

Meson and tetraquark modes are **mutually exclusive** (tetraquark wins if both are set). Scattering needs prior meson resample runs for all `Ns_list` volumes — see [docs/RUNNING.md](docs/RUNNING.md).

---

## Example Results

Per-ensemble figures below use **\(L=12\)** (`L12M420_EV170`) as examples. All outputs live in [`result/Tcccc6600/`](result/Tcccc6600/) (default **`plot_format="png"`**). Scattering plots combine \(L=12\) and \(L=16\) into `K_s_scattering.*` / `kcot_scattering.*`. Meson \(Z_n\) / dispersion figures require a separate meson-mode run.

### GEVP (before / after diagonalization)

<p align="center">
  <img src="result/Tcccc6600/GEVP_before_L12M420_EV170.png" alt="GEVP matrix before diagonalization" width="48%" />
  <img src="result/Tcccc6600/GEVP_after_L12M420_EV170.png" alt="GEVP matrix after diagonalization" width="48%" />
</p>

### GEVP eigenvectors

<p align="center">
  <img src="result/Tcccc6600/GEVP_eigenvector_L12M420_EV170.png" alt="GEVP eigenvectors" width="55%" />
</p>

### Effective mass \(E_n\)

<p align="center">
  <img src="result/Tcccc6600/En_tetraquark_L12M420_EV170.png" alt="Tetraquark effective mass" width="48%" />
  <img src="result/Tcccc6600/En_meson_L12M420_EV170.png" alt="Meson effective mass" width="48%" />
</p>

### Overlap factors \(Z_n/Z_0\) and dispersion \(E_n^2\)

<p align="center">
  <img src="result/Tcccc6600/Zn_meson_L12M420_EV170.png" alt="Overlap factors Zn" width="48%" />
  <img src="result/Tcccc6600/Dispersion_meson_L12M420_EV170.png" alt="Dispersion relation" width="48%" />
</p>

### Scattering observables (cross-ensemble: \(L=12+16\))

<p align="center">
  <img src="result/Tcccc6600/K_s_scattering.png" alt="Scattering K(s)" width="48%" />
  <img src="result/Tcccc6600/kcot_scattering.png" alt="k cot delta_0" width="48%" />
</p>

---

## Quick Start

**Requirements:** Python 3.10+, TeX (for default LaTeX labels). See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

```bash
git clone https://github.com/Geng-Li-1995/lattice_scattering.git
cd lattice_scattering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 1 — resampling (required before scattering)
python run_resample.py

# Step 2 — analysis and plotting
python main.py
```

> Raw correlators (`data/**/raw/*.npy`) and resampled files are **not** in this repository. Place them locally before running. Full setup: [docs/RUNNING.md](docs/RUNNING.md).

### Configuration

All parameters live in `input/Tcccc6600_input.py`. Key switches:

```python
lattice_Ns: int = 12
is_tetraquark_analysis: bool = True   # default workflow (mutually exclusive with meson mode)
is_gevp: bool = True
run_scattering: bool = True
plot_meff: bool = True
plot_dispersion: bool = True          # meson mode only
plot_format: str = "png"              # "png" or "pdf"
resample_type: str = "jackknife"      # or "bootstrap"
```

| \(L\) | \(N_t\) | \(m_\pi\) (MeV) | Distillation EV | Gauge configs | \(a^{-1}\) (GeV) |
|-------|---------|-----------------|-----------------|---------------|------------------|
| 12 | 96 | 420 | 170 | 400 | 7.219 |
| 16 | 128 | 420 | 120 | 400 | 7.219 |

`EV` in filenames (e.g. `L12M420_EV170`) denotes the **distillation eigenvector count**, not the number of gauge configurations. The correlator `sample` axis indexes the 400 gauge configs.

Scattering combines both volumes via `Ns_list = [12, 16]`.

---

## Data Availability

| Content | In repository? |
|---------|----------------|
| Analysis source code | Yes |
| Configuration (`input/Tcccc6600_input.py`) | Yes |
| Result figures (`result/Tcccc6600/`, format set by `plot_format`) | Yes |
| Raw correlators (`data/**/raw/*.npy`) | **No** (~80 MB) |
| Resampled energies (`data/**/resampled/*.npy`) | **No** |

| File pattern | Shape / content |
|--------------|-----------------|
| `correlation_meson_L{Ns}M{M}_EV{EV}.npy` | `[channel, momentum, time, sample]` — `sample` = 400 gauge configs; `EV` = distillation eigenvectors |
| `correlation_tetraquark_L{Ns}M{M}_EV{EV}.npy` | `[ch_src, mom_src, ch_snk, mom_snk, time, sample]` — same convention |
| `resample_En_{type}_L{Ns}M{M}_EV{EV}.npy` | Per-configuration energies (400 jackknife/bootstrap samples) |
| `resample_ksi_meson_L{Ns}M{M}_EV{EV}.npy` | Dispersion scale \(\xi\) |

---

## Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

---

## Notes

- Plotting calls `plt.show()`; on headless clusters set matplotlib backend `Agg` (see [docs/RUNNING.md](docs/RUNNING.md)).
- Draw order: error band → fit curve → data points → legend (`ZORDER_*` in `plotting/plot_set.py`). Fit bands use `fill_between` with `FIT_CURVE_ALPHA` (default **0.3**).
- Analysis modes are mutually exclusive: tetraquark wins if both meson and tetraquark flags are set. Meson \(Z_n\) / dispersion require a separate run with `is_meson_analysis=True`, `is_tetraquark_analysis=False`.
- Tetraquark + scattering needs resampled meson energies and \(\xi\) from prior meson resample runs.

---

## License

Not specified. Contact the maintainer before redistribution.
