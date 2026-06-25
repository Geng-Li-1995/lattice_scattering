# Lattice QCD Tetraquark Scattering

[![CI](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml)

Python pipeline for lattice tetraquark spectroscopy and finite-volume scattering. Ingests **6D Monte Carlo correlators** (up to \(\sim10^7\) `float64` elements per file, **400** gauge configs, \(N_t\sim10^2\)), runs GEVP, Bayesian multi-state fits, jackknife/bootstrap resampling, and Lüscher phase-shift extraction with full error propagation.

Switch systems via `BuildConfig("<System>")` in `main.py`. Physics and ensemble metadata live in `input/input_<System>.py`.

**Reference system:** Tcccc6600 — \(\eta_c\eta_c\) / \(J/\psi\,J/\psi\), \(L=12,16\), \(a_t^{-1}=7.219\) GeV, 400 configs/volume.

Raw / resampled `.npy` data are local only (`data/<system>/`).

---

## Technical capabilities

| Layer | Implementation |
|-------|----------------|
| **Linear algebra** | Generalized eigenvalue problem (`scipy.linalg.eig`); GEVP rotation via `numpy.einsum`; FVE matrix assembly from 6D tetraquark tensors |
| **Fitting** | Bayesian nonlinear least squares (`lsqfit` + `gvar`); 2-/3-state cosh models; dispersion and \(t_{\min}\) window scans; 4Q/2Q ratio fits |
| **Resampling** | Leave-one-out jackknife and bootstrap over the gauge ensemble; correlated `gvar` objects through fits and scattering |
| **Scattering** | Lüscher zeta summation on \(10^5\)-point grids; rest-frame and moving-frame kinematics; \(K(s)\) and \(k\cot\delta_0\) extraction |
| **Parallelism** | `joblib` parallel zeta-table generation; vectorized NumPy throughout; cached `.npy` artefacts (`resampled/`, `data/zeta/`) |
| **Software** | Typed dataclass wrappers (`Correlator4D`, `Config`); registry-based fit models; switch-controlled pipeline stages; LaTeX publication figures |
| **CI / tests** | `pytest` on Python 3.10 & 3.12 without lattice data; GitHub Actions on every push to `main` |

**Stack:** Python 3.10+ · NumPy · SciPy · gvar · lsqfit · joblib · Matplotlib · pytest

---

## Project structure

```
lattice_scattering/
├── main.py                      # entrypoint: resample → GEVP → fits → plots → scattering
├── input/
│   ├── config.py                # Config, BuildConfig, SelectorType, ensemble tags
│   ├── input_Tcccc6600.py       # InputControl switches + ENSEMBLE_DB
│   ├── input_X3872.py
│   └── input_Zc3900.py
├── data/
│   ├── correlators.py           # Correlator4D, TetraquarkCorrelator, AnalysisCorrelators
│   └── io.py                    # read_raw_files, read_resampled_files, MF scatter cache
├── analysis/
│   ├── gevp.py                  # FVE matrix build, generalized eig, diagonal extraction
│   ├── fit_mass.py              # RunFitting, En / dispersion lookups
│   ├── fit_tmin.py              # t_min scan, 4Q/2Q ratio series and fits
│   ├── scattering.py            # zeta tables, K(s) / kcot fits, rest + moving frame
│   └── models.py                # cosh / ratio models, priors, MODEL_REGISTRY
├── statistics/
│   ├── jackknife.py / bootstrap.py
│   └── resample.py              # run_resample_statistics()
├── plotting/
│   ├── plot_set.py              # RC params, colors, save_figure()
│   ├── plot_gevp.py / plot_mass.py / plot_tmin.py / plot_scattering.py
├── tests/                       # synthetic-array unit tests (see docs/TESTING.md)
└── docs/                        # RUNNING.md, TESTING.md, DEPENDENCIES.md
```

### Module map

| Stage | Module | Key output |
|-------|--------|------------|
| Load correlators | `data/io.py` | `RawCorrelators` |
| Shape-safe arrays | `data/correlators.py` | `Correlator4D`, `TetraquarkCorrelator`, `AnalysisCorrelators` |
| GEVP | `analysis/gevp.py` | 4D tetraquark correlators, eigenvectors |
| Effective mass | `analysis/fit_mass.py`, `models.py` | `en_fit_list`, `disp_fit_list` |
| \(t_{\min}\) + ratio | `analysis/fit_tmin.py`, `plotting/plot_tmin.py` | per-window \(E\), \(\chi^2\)/d.o.f. |
| Resampling | `statistics/resample.py` | `data/<system>/resampled/*.npy` |
| Scattering | `analysis/scattering.py` | \(K(s)\), \(k\cot\delta_0\) per jackknife sample |
| Figures | `plotting/plot_*.py` | `result/<system>/*.{png,pdf}` |

Configuration, I/O, analysis, statistics, and plotting are decoupled. A new system needs only `input/input_<System>.py`, local `data/<System>/` arrays, and a one-line change in `main.py`.

### Runtime data objects

| Symbol | Type / shape |
|--------|----------------|
| `raw` | `RawCorrelators` — meson `[ch, mom, t, sample]`; tetraquark raw `[ch_src, mom_src, ch_snk, mom_snk, t, sample]` |
| `corr` | `AnalysisCorrelators` — both branches 4D after GEVP |
| `resampled` | Per-config `En` and meson \(\xi\) from jackknife/bootstrap |
| `en_fit_list` | Effective-mass fit results keyed by `(channel, n^2)` |
| `scattering_dict` | Scattering observables per ensemble volume |

`mom` is the momentum quantum number \(n^2\) used as an **array index** (from `chan_momt_list`).

### `main.py` execution order

1. Build `Config` from `input/input_<System>.py`
2. `run_resample_statistics()` — if `run_resample=True` (jackknife/bootstrap over raw data)
3. Meson branch — if `is_meson_analysis=True` (fits; t_min plots skipped on meson-only pass)
4. Tetraquark branch — if `is_tetraquark_analysis=True`: GEVP → En fits → optional `plot_tmin_workflow()`
5. `run_scattering_analysis()` — if `run_scattering=True` (reads `resampled/`)

---

## Pipeline

```
data/<system>/raw/*.npy
       │
       ▼  main.py
  run_resample_statistics()   →  resampled/*.npy   (400× jackknife when enabled)
  process_GEVP()              →  4D tetraquark correlators
  effective_mass()            →  Bayesian En, Zn, dispersion
  plot_tmin_workflow()        →  optional t_min + ratio stability plots
  run_scattering_analysis()   →  K(s), k cot δ₀
       │
       ▼
  result/<system>/*.{png,pdf}
```

---

## Supported systems

| System | Input | Ensembles | Channels |
|--------|-------|-----------|----------|
| `Tcccc6600` | `input/input_Tcccc6600.py` | \(L=12,16\), EV 170/120 | \(\eta_c\eta_c\), \(J/\psi J/\psi\) |
| `X3872` | `input/input_X3872.py` | \(L=16\), EV 70 | \(\pi J/\psi\), \(\rho\eta_c\), \(DD^*\), \(D^*D^*\) |
| `Zc3900` | `input/input_Zc3900.py` | \(L=16\), EV 70 | same as X3872 |

---

## Data scale

All arrays on disk: **`float64`**. Typical gauge ensemble: **400** configs on the `sample` axis.

### Tcccc6600 raw files

| File | Shape | Elements | Size |
|------|-------|----------|------|
| `correlation_meson_L12M420_EV170.npy` | `[2, 10, 96, 400]` | 768 k | 5.9 MiB |
| `correlation_tetraquark_L12M420_EV170.npy` | `[2, 5, 2, 5, 96, 400]` | 3.84 M | 29 MiB |
| `correlation_meson_L16M420_EV120.npy` | `[2, 10, 128, 400]` | 1.02 M | 7.8 MiB |
| `correlation_tetraquark_L16M420_EV120.npy` | `[2, 5, 2, 5, 128, 400]` | 5.12 M | 39 MiB |

Tetraquark raw rank is **6D** vs **4D** for meson — dominant memory and GEVP tensor cost. Both \(L=12+16\) volumes: **~82 MiB** raw input.

### Resampled cache (after jackknife)

| Pattern | Shape (example) | Role |
|---------|-----------------|------|
| `resample_En_meson_*.npy` | `[6, 5, 401]` | Per-config meson energies |
| `resample_En_tetraquark_*.npy` | `[1, 3, 401]` | Tetraquark levels for scattering |
| `resample_ksi_meson_*.npy` | `[6, 401]` | Dispersion scale \(\xi\) |

Scattering reads **401-sample** energy vectors — no reload of raw multi‑MiB correlators.

### Jackknife cost (400 configs)

| Stage | Single pass | Jackknife |
|-------|-------------|-----------|
| GEVP + En fits | 1× | **400×** |
| Meson dispersion | 1× | **400×** |
| Scattering on `resampled/` | 1× | 1× |

---

## Scientific context

Fully-charm tetraquarks \(T_{cc\bar{c}\bar{c}}\) are exotic-hadron candidates seen at the LHC. Lattice QCD provides a first-principles route to \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) interactions: finite-volume energies from Monte Carlo correlators are mapped to scattering amplitudes via the **Lüscher formalism**, after GEVP removes operator mixing between channels.

Key physics results on Tcccc6600:

- First lattice QCD evidence for a **\(2^{++}\) resonance** in \(J/\psi\,J/\psi\) near **6.6 GeV**, compatible with the broad **\(X(6600)\)** structure reported by ATLAS and CMS ([Nature **648**, 58 (2025)](https://www.nature.com/articles/s41586-025-09278-2); [arXiv:2506.07944](https://arxiv.org/abs/2506.07944)).
- Preferred \(J^{PC}=2^{++}\) assignment consistent with the CMS angular analysis.
- Separate **\(0^{++}\)** and **\(2^{++}\)** scattering amplitudes extracted from the same GEVP-diagonalized spectrum.

| Step | Method | Observable |
|------|--------|------------|
| Operator mixing | GEVP on \(\eta_c\eta_c\)–\(J/\psi\,J/\psi\) matrix | Physical FVE levels |
| Level extraction | Multi-state Bayesian cosh fit | \(E_n\), \(Z_n\) |
| Scale setting | Meson dispersion relation | lattice spacing \(\xi\) |
| Scattering | Lüscher zeta function | \(K(s)\), \(k\cot\delta_0(s)\) |
| Errors | Jackknife / bootstrap (400 configs) | Correlated fit and scattering errors |

**Scattering interpretation:** \(K(s)\) is built from finite-volume energy shifts and encodes the S-wave interaction strength as a function of centre-of-mass energy squared \(s\). \(k\cot\delta_0(s)\) is the standard Lüscher relation to the S-wave phase shift \(\delta_0\). **Zeros of \(K(s)\) correspond to poles of the S-matrix** — lattice predictions of resonances or bound states that can be compared with experimental enhancements such as \(X(6600)\) in the \(J/\psi\,J/\psi\) channel.

---

## Example results (Tcccc6600)

Spectroscopy → fit stability → scattering. Default figures: \(L=12\) (`L12M420_EV170`); \(t_{\min}\) scan: \(L=16\).

### GEVP — operator mixing

<p align="center">
  <img src="result/Tcccc6600/GEVP_before_L12M420_EV170.png" alt="GEVP before" width="48%" />
  <img src="result/Tcccc6600/GEVP_after_L12M420_EV170.png" alt="GEVP after" width="48%" />
</p>

### Effective mass \(E_n\)

<p align="center">
  <img src="result/Tcccc6600/En_meson_L12M420_EV170.png" alt="Meson En" width="48%" />
  <img src="result/Tcccc6600/En_tetraquark_L12M420_EV170.png" alt="Tetraquark En" width="48%" />
</p>

### \(t_{\min}\) scan — \(J/\psi\,J/\psi\), \(n^2=0,1\)

<p align="center">
  <img src="result/Tcccc6600/E2_mom0_tmin_ratio_L16M420_EV120.png" alt="tmin E2 mom0" width="48%" />
  <img src="result/Tcccc6600/E2_mom1_tmin_ratio_L16M420_EV120.png" alt="tmin E2 mom1" width="48%" />
</p>

### Scattering — \(K(s)\) and \(k\cot\delta_0\) (\(L=12+16\))

Lüscher extraction from jackknife-resampled finite-volume energies on **both** \(L=12\) and \(L=16\). The combined fit uses meson and tetraquark levels from GEVP together with the dispersion-calibrated scale \(\xi\).

| Figure | Physical content |
|--------|------------------|
| **`K_s_scattering.png`** | \(K(s)\) vs \(s=m_{\rm CM}^2\): interaction strength in the \(J/\psi\,J/\psi\) S-wave channel. **A zero crossing of \(K(s)\) signals an S-matrix pole** — a candidate resonance or bound state. The tensor (\(2^{++}\)) and scalar (\(0^{++}\)) sectors are resolved from the GEVP level ordering. |
| **`kcot_scattering.png`** | \(k\cot\delta_0(s)\) vs \(k^2\): equivalent phase-shift parameterization. Phase-shift structure near threshold and above reflects the same underlying pole content as \(K(s)\). |

These lattice zeros/poles are compared with the experimental **\(X(6600)\)** enhancement in double-\(J/\psi\) production: agreement in position and preferred \(J^{PC}\) supports a genuine near-threshold exotic structure rather than a kinematic artefact.

<p align="center">
  <img src="result/Tcccc6600/K_s_scattering.png" alt="K(s)" width="48%" />
  <img src="result/Tcccc6600/kcot_scattering.png" alt="k cot delta" width="48%" />
</p>

---

## Output naming

Extension from `plot_format` (`png` / `pdf`). Tag: `L{Ns}M{M}_EV{EV}` (`EV` = distillation eigenvectors).

| Plot | Pattern |
|------|---------|
| GEVP | `GEVP_{before,after,eigenvector}_{tag}.{ext}` |
| Effective mass | `En_{meson,tetraquark}_{tag}.{ext}` |
| Overlap / dispersion | `Zn_meson_*`, `Dispersion_meson_*` |
| \(t_{\min}\) scan | `E{n}_mom{n2}_tmin_{tag}.{ext}` or `*_tmin_ratio_*` |
| Scattering | `K_s_scattering.{ext}`, `kcot_scattering.{ext}` |

---

## Quick start

```bash
git clone https://github.com/Geng-Li-1995/lattice_scattering.git
cd lattice_scattering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

MPLBACKEND=Agg pytest
python main.py
```

[docs/RUNNING.md](docs/RUNNING.md) · [docs/TESTING.md](docs/TESTING.md) · [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)

### Key switches (`input/input_<System>.py`)

```python
lattice_Ns: int = 12
is_meson_analysis: bool = True
is_tetraquark_analysis: bool = True
is_gevp: bool = True
run_tmin: bool = True
is_ratio: bool = True
run_resample: bool = False      # True → 400× jackknife resampling pass
run_scattering: bool = True
plot_format: str = "png"
resample_type: str = "jackknife"
```

---

## Tests & CI

GitHub Actions: `pytest` on Python 3.10 & 3.12, no lattice `.npy` required ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

---

## License

Not specified. Contact the maintainer before redistribution.
