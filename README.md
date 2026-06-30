# Lattice QCD Tetraquark Scattering

[![CI](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml)

Production-grade lattice QCD pipeline for **tetraquark spectroscopy** and **finite-volume scattering**: six-dimensional correlators → GEVP → multi-state Bayesian fits → ratio methods → Lüscher phase shifts, with **correlated jackknife uncertainties** propagated end-to-end.

| | |
|--|--|
| **Systems** | `Tcccc6600` · `X3872` · `Zc3900` — one `input_<System>.py` per physics setup |
| **Reference** | Fully-charm \(T_{cc\bar c\bar c}\): \(\eta_c\eta_c\) / \(J/\psi\,J/\psi\), \(L=12{+}16\), \(a_t^{-1}=7.219\) GeV |
| **Scale** | 6D raw tensors · **401** jackknife replicas · \(10^5\)-point Lüscher \(\zeta\) tables |
| **Data** | Local `data/<system>/` only (not in git) · figures in `result/<system>/` |

---

## Contents

1. [Technical overview](#technical-overview)
2. [Architecture](#architecture)
3. [Analysis capabilities](#analysis-capabilities)
4. [Software engineering](#software-engineering)
5. [Physics (Tcccc6600)](#physics-tcccc6600)
6. [Example results](#example-results)
7. [Data contract](#data-contract)
8. [Quick start](#quick-start)
9. [Configuration](#configuration)
10. [Tests & CI](#tests--ci)
11. [Publications](#publications)

---

## Technical overview

This repository implements a **modular, switch-driven** analysis chain used in lattice studies of exotic hadrons. Design goals:

- **Correct tensor geometry** — typed 4D/6D correlators; no silent axis swaps between meson and tetraquark branches.
- **Unified uncertainty** — `gvar` + jackknife/bootstrap through fits, ratio series, and scattering observables.
- **Label-driven configuration** — channel indices resolved from LaTeX names in `ENSEMBLE_DATABASE`, not hard-coded in analysis code.
- **Decoupled stages** — spectroscopy and scattering can run independently; scattering replays from cached `resampled/` energies.

### Computational scale (Tcccc6600)

| Quantity | Value |
|----------|-------|
| Raw tetraquark rank | **6D** `[chan_src, mom_src, chan_snk, mom_snk, time, sample]` |
| Raw file size | ~30–40 MiB / volume (\(\sim10^6\)–\(10^7\) `float64`) |
| Jackknife replicas | **401** on the `sample` axis |
| Time extent \(N_t\) | 96 (\(L=12\)) / 128 (\(L=16\)) |
| Resampling cost | **401×** GEVP + fit per volume when `run_resample_analysis=True` |
| Lüscher grid | \(10^5\) \(q^2\) points, built once (`joblib` parallel) |

Meson correlators are **4D**; tetraquark raw data dominates memory. After GEVP both branches collapse to 4D for fitting.

**Stack:** Python 3.10+ · NumPy · SciPy · gvar · lsqfit · joblib · Matplotlib · pytest

---

## Architecture

### Module layers

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **Configuration** | `input/` | `InputControl` switches, `ENSEMBLE_DATABASE`, `BuildConfig` → `Config` |
| **Data** | `data/` | Typed correlators (`Correlator4D` / 6D tetraquark), `io.py` path helpers, `raw/` & `resampled/` |
| **Analysis** | `analysis/` | GEVP · effective mass · t_min & ratio · Lüscher scattering · `models.py` registry |
| **Statistics** | `statistics/` | Jackknife / bootstrap, `run_resample_statistics` → per-replica energies |
| **Plotting** | `plotting/` | Shared `plot_set.py` style; GEVP, \(E_n\), ratio, t_min, scattering figures |

### Pipeline (`main.py`)

```
data/<system>/raw/*.npy  [chan, mom, time, sample=401]
        │
        ▼
input/input_<System>.py  →  BuildConfig  →  Config
        │
        ├─ [optional] statistics/resample.py     →  data/<system>/resampled/*.npy
        │
        ├─ meson branch
        │     analysis/fit_mass.py             →  En, Z_n, dispersion
        │     plotting/plot_mass.py            →  En_*, Zn, Dispersion
        │     plotting/plot_tmin.py            →  En_tmin_meson*  (if enabled)
        │
        ├─ tetraquark branch
        │     analysis/gevp.py                 →  4D correlators
        │     analysis/fit_mass.py             →  En, Z_n
        │     plotting/plot_gevp.py            →  GEVP_*
        │     plotting/plot_ratio.py           →  Ratio_*  (if enabled)
        │     plotting/plot_tmin.py            →  En_tmin_tetraquark*  (if enabled)
        │
        └─ scattering (from resampled/, no raw reload)
              analysis/scattering.py           →  K(s), k cot δ₀
              plotting/plot_scattering.py      →  K_s_scattering, kcot_scattering
        │
        ▼
result/<system>/*.{png,pdf}
```

Meson and tetraquark switches are **independent**. Ratio / t_min on tetraquark loads meson raw correlators when `run_ratio_analysis=True`.

### Uncertainty propagation

| Stage | Method |
|-------|--------|
| Effective mass | `lsqfit` + `gvar` priors; jackknife slices via `statistics/jackknife.py` |
| Ratio \(R_n(t)\) | Leave-one-out correlators → ratio series → jackknife (`ratio_series_mean_err`) |
| Resampled energies | Per-replica `En`, \(\xi\) written to `resampled/` for scattering |
| Scattering | \(K(s)\), \(k\cot\delta_0\) evaluated **per jackknife sample**, then combined |
| Plots | Error bands from `gvar` mean ± sdev; t_min panels use ±10× reference error |

---

## Analysis capabilities

### 1 · Generalized eigenvalue problem (GEVP)

Tetraquark operators are coupled in a finite-volume basis (FVE). GEVP diagonalizes this coupling so each eigenlevel yields a **single-channel** correlator suitable for multi-state cosh fits.

**Physics.** For operators \(O_\alpha\) at reference times \(t_0 < t_1\), solve

\[
C(t_1)\,\psi = \lambda\, C(t_0)\,\psi,
\]

with \(C_{\alpha\beta}(t)=\langle O_\alpha(t)\,O_\beta^\dagger(0)\rangle\). Eigenvectors \(v_\beta\) rotate the matrix; **diagonal** elements \(C'_{\beta\beta}(t)\) are the optimized levels fed into 3-state fits.

**Pipeline** (`analysis/gevp.py`, switch `run_GEVP_analysis`):

| Step | What happens |
|------|----------------|
| **6D → FVE matrix** | Raw `[chan_src, mom_src, chan_snk, mom_snk, time, sample]` blocks are assembled into \(C_{\alpha\beta}(t)\). Index \(\alpha\) runs over all \((\text{chan}, n^2)\) pairs in `channel_momentum_list` (Tcccc6600 \(L=12\): 3 + 5 = **8** FVE rows). Shape: `[n_fve, n_fve, time, sample]`. |
| **Solve** | `scipy.linalg.eig` on ensemble-averaged \(C(t_1)\) and \(C(t_0)\). Times from `ENSEMBLE_DATABASE["GEVP"]` → `Config.t_GEVP = (t_0, t_1, t_\mathrm{sort})`. Example Tcccc6600: \(L=12\) → `(15, 25, 20)`; \(L=16\) → `(30, 40, 35)`. |
| **Sort** | `greedy_sort_eigenvectors` — greedy matching of large \(|v_{\alpha\beta}|\) so eigenlevel order is stable across jackknife samples. |
| **Rotate** | \(C'(t) = v^\dagger C(t) v\) via `numpy.einsum("ai,abxy,bj->ijxy", ...)`. |
| **Extract levels** | Diagonal \(C'_{\beta\beta}(t)\) → 4D `[chan, mom, time, sample]` for downstream fits (`process_GEVP` → `AnalysisCorrelators.with_tetraquark`). |

If `run_GEVP_analysis=False`, the FVE matrix is passed through without rotation (diagonal extraction only).

**Figures** (`plotting/plot_gevp.py`, `run_GEVP_analysis=True`):

| Output | Content |
|--------|---------|
| `GEVP_before_{tag}` | Normalized \(\|C_{\alpha\beta}(t_\mathrm{sort})\|\) **before** rotation; operator mixing visible off-diagonal. |
| `GEVP_after_{tag}` | Same normalization **after** rotation; near-diagonal — levels decoupled. |
| `GEVP_eigenvector_{tag}` | Real part of \(v_\beta^{(n)}\): eigenvector component vs. eigenlevel index \(\beta\). |

Matrix plots: normalized \(C_{\alpha\beta}/\sqrt{C_{\alpha\alpha}C_{\beta\beta}}\) at \(t_\mathrm{sort}\); black squares mark the diagonal. Axis labels list each FVE operator, e.g. \(\eta_c\eta_c(1)\), \(J/\psi J/\psi(0)\).

**Config:** `t_GEVP` per ensemble in `ENSEMBLE_DATABASE`; `run_GEVP_analysis` only on the tetraquark branch.

### 2 · Spectroscopy (effective mass)

| Item | Implementation |
|------|----------------|
| Models | Registry-based cosh: 2-state (meson) / 3-state (tetraquark) (`models.py`) |
| Fit | `lsqfit.nonlinear_fit` with channel/momentum priors from `ENSEMBLE_DATABASE` |
| Observables | \(E_n\), \(Z_n/Z_0\), dispersion \(E_n^2(n^2)\) → lattice scale \(\xi\) |
| Figures | Multi-channel \(E_n\) overlay (`plot_mass.py`); `FIG_WIDE` layout |

### 3 · Ratio method & \(t_{\min}\) stability

| Item | Implementation |
|------|----------------|
| Shifted ratio (`is_ratio_shift=True`) | \(R_n(t+a_t)=\Delta C_4/\Delta C_2^2\) with \(\Delta\) over \(2a_t\) lattice steps; identical mesons share one \(C_2\) |
| Direct ratio (`is_ratio_shift=False`) | \(R(t)=C_4(t)/[C_a(t)\,C_b(t)]\) (open-charm / distinguishable mesons) |
| t_min scan | Symmetric window scan; meson 2-state / tetraquark 3-state + optional ratio overlay |
| Combine | Reference energy band at configured \(t_{\min}\) on all t_min figures |
| Figures | `Ratio_{tag}` (fit-window y-limits); `En_tmin_{branch}{n}_mom{m}_{tag}` (`plot_ratio.py`, `plot_tmin.py`) |

### 4 · Finite-volume scattering

| Item | Implementation |
|------|----------------|
| Framework | Lüscher quantization condition; precomputed \(\zeta(q^2)\) under `data/zeta/` |
| Observables | \(K(s)=\sqrt{s}/(k\cot\delta_0)\); phase shift from \(k\cot\delta_0(k^2)\) |
| Frames | Rest frame; **moving frame** (`run_MF_analysis`, `MF_d_vec`, X3872) |
| Multi-\(L\) | `scattering_Ns_mom` selects momentum subset per spatial size |
| Fits | `Ks_linear` or `kcot_quadratic` (`scattering_fit_mode`) |
| Figures | \(K(s)\) vs \(s\); \(k\cot\delta_0\) vs \(k^2\) with below-threshold \(ik\) guide (`plot_scattering.py`) |

### 5 · Resampling & I/O

| Item | Implementation |
|------|----------------|
| Jackknife | Leave-one-out on `sample` axis (401 replicas) |
| Bootstrap | Optional (`resample_type`, `n_boot`) |
| Paths | Centralized in `data/io.py`; ensemble tags `L{Ns}M{M}_EV{EV}` |
| Types | `Correlator4D`, `TetraquarkCorrelator`, `AnalysisCorrelators` — shape-safe `.at()` |

---

## Software engineering

### Configuration model

```
input/input_<System>.py
├── InputControl          # user switches (3 sections: mass · scattering · plot)
├── ENSEMBLE_DATABASE           # per-ensemble meson/tetraquark blocks
└── get_lattice_params()  # lattice_Ns → (Ns, Nt, M_π, EV)

BuildConfig("<System>").build_config_from_control("meson" | "tetraquark")
    → frozen Config dataclass
```

Branch-gated switches (`_branch_switches`): e.g. dispersion only on meson config; GEVP/ratio only on tetraquark.

### Channel resolution (by label fields)

Physics channels are matched from **string labels**, not array indices:

| Field | Resolves to |
|-------|-------------|
| `channel_name_list` / `channel_momentum_list` | Chan axis and \(n^2\) values per ensemble |
| `scattering_channel` | `ScatteringChanMatch` → meson a, meson b, tetra indices |
| `scattering_channel_MF` | Moving-frame tetra channel (defaults to rest) |
| `ratio_point_by_label(config, label, n²)` | `ChanMom(chan, mom)` for ratio / t_min lookup |

Tetraquark labels such as `r"J/\psi\,J/\psi"` split on `\,` into meson tokens; normalization ignores separators when matching `scattering_channel`.

### Extensibility

| Task | Steps |
|------|-------|
| New hadron system | Copy `input_<System>.py` → add `data/<System>/raw/` → `BuildConfig("System")` in `main.py` |
| New fit model | Register in `analysis/models.py` → `MODEL_REGISTRY` |
| New plot | Subclass `BasePlotter` in `plotting/plot_set.py` |

### Code layout

```
lattice_scattering/
├── main.py
├── input/          config.py · input_{Tcccc6600,X3872,Zc3900}.py
├── data/           correlators.py · io.py · <System>/{raw,resampled}/ · zeta/
├── analysis/       gevp · fit_mass · fit_tmin · scattering · models
├── statistics/     jackknife · bootstrap · resample
├── plotting/       plot_set · plot_gevp · plot_mass · plot_ratio · plot_tmin · plot_scattering
├── result/<System>/    example figures (tracked)
├── docs/           RUNNING · TESTING · DEPENDENCIES
└── tests/          ~19 consolidated unit tests (synthetic data only)
```

---

## Physics (Tcccc6600)

Fully-charm tetraquarks \(T_{cc\bar{c}\bar{c}}\) are exotic-hadron candidates at the LHC. This analysis chain targets \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) scattering and comparison with LHC data.

**Key result:** a **\(2^{++}\)** candidate in \(J/\psi\,J/\psi\) near **6.6 GeV**, compatible with **\(X(6600)\)** ([Nature **648**, 58 (2025)](https://www.nature.com/articles/s41586-025-09278-2); [arXiv:2506.07944](https://arxiv.org/abs/2506.07944)), with separate **\(0^{++}\)** and **\(2^{++}\)** amplitudes from one GEVP spectrum.

**Scattering observables:** \(K(s)\) vs \(s=m_{\rm CM}^2\) encodes S-wave interaction strength; \(k\cot\delta_0\) gives the phase shift \(\delta_0\). **Zeros of \(K(s)\) are S-matrix poles** — resonance or bound-state positions to compare with experimental enhancements such as \(X(6600)\) in double-\(J/\psi\) production.

---

## Example results

\(L=12\) (`L12M420_EV170`) unless noted; \(t_{\min}\) scans on \(L=16\) (`L16M420_EV120`).

### Generalized eigenvalue problem (GEVP)

Generalized eigenvalue problem on the tetraquark FVE matrix \(C_{\alpha\beta}(t)\) (\(t_0=15\), \(t_1=25\), plot at \(t_\mathrm{sort}=20\) for \(L=12\)). **Before:** strong operator mixing between \(\eta_c\eta_c\) and \(J/\psi\,J/\psi\) blocks. **After:** near-diagonal matrix — optimized levels for 3-state fits. **Eigenvector:** composition of each GEVP level in the original operator basis.

<p align="center">
  <img src="result/Tcccc6600/GEVP_before_L12M420_EV170.png" alt="GEVP matrix before diagonalization" width="48%" />
  <img src="result/Tcccc6600/GEVP_after_L12M420_EV170.png" alt="GEVP matrix after diagonalization" width="48%" />
</p>

<p align="center"><sub><code>GEVP_before_L12M420_EV170</code> · <code>GEVP_after_L12M420_EV170</code> — normalized \(C_{\alpha\beta}(t/a_t=20)\), snk/src = FVE operators</sub></p>

Eigenvectors \(v_\beta^{(n)}\) (eigenlevel \(n\), component \(\beta\)):

<p align="center">
  <img src="result/Tcccc6600/GEVP_eigenvector_L12M420_EV170.png" alt="GEVP eigenvector" width="48%" />
</p>

<p align="center"><sub><code>GEVP_eigenvector_L12M420_EV170</code></sub></p>

### Effective mass \(E_n\)

<p align="center">
  <img src="result/Tcccc6600/En_meson_L12M420_EV170.png" alt="Meson En" width="48%" />
  <img src="result/Tcccc6600/En_tetraquark_L12M420_EV170.png" alt="Tetraquark En" width="48%" />
</p>

### Ratio \(R_n(t/a_t)\)

Shifted-ratio series \(\Delta C_4/\Delta C_2^2\) (data) and reference-window fit (band + curve). **Left:** \(L=12\); **right:** \(L=16\).

<p align="center">
  <img src="result/Tcccc6600/Ratio_L12M420_EV170.png" alt="Ratio L12" width="48%" />
  <img src="result/Tcccc6600/Ratio_L16M420_EV120.png" alt="Ratio L16" width="48%" />
</p>

<p align="center"><sub><code>Ratio_L12M420_EV170</code> · <code>Ratio_L16M420_EV120</code></sub></p>

### \(t_{\min}\) scan — meson

2-state ○ + Combine. \(\eta_c\) / \(J/\psi\), \(n^2=0\), \(L=16\).

<p align="center">
  <img src="result/Tcccc6600/En_tmin_meson0_mom0_L16M420_EV120.png" alt="meson tmin eta_c" width="48%" />
  <img src="result/Tcccc6600/En_tmin_meson1_mom0_L16M420_EV120.png" alt="meson tmin J/psi" width="48%" />
</p>

### \(t_{\min}\) scan — tetraquark (\(J/\psi\,J/\psi\), \(n^2=0,1\))

3-state ○, ratio ×, Combine.

<p align="center">
  <img src="result/Tcccc6600/En_tmin_tetraquark1_mom0_L16M420_EV120.png" alt="tetra tmin mom0" width="48%" />
  <img src="result/Tcccc6600/En_tmin_tetraquark1_mom1_L16M420_EV120.png" alt="tetra tmin mom1" width="48%" />
</p>

### Scattering (\(L=12+16\))

Left: \(K(s)=\sqrt{s}/(k\cot\delta_0)\) vs. invariant mass squared \(s\); right: \(k\cot\delta_0\) vs. \(k^2\) (dashed \(ik\) guide below threshold). Points combine rest and moving frames on \(L=12\) and \(L=16\); bands show jackknife uncertainty on the fitted curves.

In the Lüscher quantization condition, **zeros of \(K(s)\) are poles of the S-matrix** — bound states or resonances in the \(J/\psi\,J/\psi\) S-wave channel. A zero near \(s\simeq(6.6\,\mathrm{GeV})^2\) is not a kinematic artifact of two free charmonia passing through threshold; it marks an **interacting state whose internal structure goes beyond meson–meson factorization**. Together with the GEVP spectrum above, such a pole supports a **genuinely new fully-charm tetraquark configuration** (\(cc\bar{c}\bar{c}\)): a compact multiquark degree of freedom distinct from the non-interacting \(J/\psi\,J/\psi\) continuum. This is precisely what collider experiments search for in double-\(J/\psi\) production — narrow or broad enhancements such as **\(X(6600)\)** — where a pole/zero in \(K(s)\) translates into an excess above the smooth two-body background. On the \(k\cot\delta_0\) panel, a sub-threshold bound state appears as the curve meeting the \(ik\) branch; agreement between \(L=12\) and \(L=16\) constrains whether the feature is a stable bound state or a finite-volume resonance.

<p align="center">
  <img src="result/Tcccc6600/K_s_scattering.png" alt="K(s)" width="48%" />
  <img src="result/Tcccc6600/kcot_scattering.png" alt="k cot delta" width="48%" />
</p>

<p align="center"><sub><code>K_s_scattering</code> · <code>kcot_scattering</code> — \(J/\psi\,J/\psi\) scattering, combined \(L=12\) + \(L=16\)</sub></p>

---

## Data contract

All arrays: **`float64`**. **`sample` = 401** jackknife replicas. **`mom`** indexes \(n^2\). **`EV`** in filenames = distillation eigenvectors.

### Tensor shapes

| Object | Axes |
|--------|------|
| Meson raw | `[chan, mom, time, sample]` |
| Tetraquark raw | `[chan_src, mom_src, chan_snk, mom_snk, time, sample]` |
| After GEVP | `[chan, mom, time, sample]` |
| Resampled `En` | `[chan, mom, sample]` |

### Supported systems

| System | \(L\) | EV | Tetraquark channels | Workflow |
|--------|-------|-----|---------------------|----------|
| **Tcccc6600** | 12, 16 | 170 / 120 | \(\eta_c\eta_c\), \(J/\psi J/\psi\) | Full: GEVP · ratio · t_min · scattering |
| **X3872** | 16 | 70 | \(\chi_{c1}\), \(DD^*\), \(J/\psi\omega\) | GEVP · MF scattering (`kcot_quadratic`) |
| **Zc3900** | 16 | 70 | \(\pi J/\psi\), \(\rho\eta_c\), \(DD^*\), \(D^*D^*\) | Scattering-first (spectroscopy optional) |

### Raw sizes (Tcccc6600)

| File | Shape | Size |
|------|-------|------|
| `correlation_meson_L12M420_EV170.npy` | `[2, 10, 96, 401]` | 5.9 MiB |
| `correlation_tetraquark_L12M420_EV170.npy` | `[2, 5, 2, 5, 96, 401]` | 29 MiB |
| `correlation_meson_L16M420_EV120.npy` | `[2, 10, 128, 401]` | 7.9 MiB |
| `correlation_tetraquark_L16M420_EV120.npy` | `[2, 5, 2, 5, 128, 401]` | 39 MiB |

Resampled energies under `data/<system>/resampled/` enable scattering **without** reloading raw correlators.

---

## Quick start

```bash
git clone https://github.com/Geng-Li-1995/lattice_scattering.git
cd lattice_scattering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
MPLBACKEND=Agg pytest && python main.py
```

Place lattice data under `data/<System>/` before running. See [docs/RUNNING.md](docs/RUNNING.md) · [docs/TESTING.md](docs/TESTING.md) · [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

---

## Configuration

### Control switches (`input/input_<System>.py`)

```python
# Effective mass
lattice_Ns = 12
run_meson_analysis = True
run_tetraquark_analysis = True
run_GEVP_analysis = True
run_tmin_analysis = True
run_ratio_analysis = True       # loads meson raw for denominators
is_ratio_shift = True           # True: R_n from ΔC4/ΔC2²; False: R = C4/(Ca·Cb)
ratio_at = 1
run_resample_analysis = False   # True → 401× jackknife write-out

# Scattering
run_scattering_analysis = True
scattering_channel = r"J/\psi\,J/\psi"
scattering_Ns_mom = {12: [0, 1, 2], 16: [0, 1]}
scattering_fit_mode = "Ks_linear"   # or "kcot_quadratic"
run_MF_analysis = False             # moving frame (X3872)

# Plot
plot_format = "png"                 # or "pdf"
is_plot_title = True
is_plot_show = True
k_sq_plot_range = (-0.25, 1.75, -15.0, 15.0)
s_plot_range = (37, 45, -9.0, 6.0)
```

### Figure naming (tag = `L{Ns}M{M}_EV{EV}`)

| Plot | Pattern |
|------|---------|
| GEVP | `GEVP_{before,after,eigenvector}_{tag}` |
| \(E_n\) | `En_{meson,tetraquark}_{tag}` |
| Ratio | `Ratio_{tag}` |
| \(t_{\min}\) | `En_tmin_{meson,tetraquark}{n}_mom{m}_{tag}` |
| Scattering | `K_s_scattering`, `kcot_scattering` |

---

## Tests & CI

| File | Coverage |
|------|----------|
| `test_config.py` | Channel labels (3 systems), ratio lookup, branch/plot switches |
| `test_fit_tmin.py` | Ratio series, fits, jackknife errors |
| `test_scattering.py` | Lüscher algebra, momentum indices, MF rows |
| `test_io.py` · `test_jackknife.py` · `test_plot.py` · `test_fit_mass.py` | I/O, resampling, plot helpers, dispersion |

**19** pytest cases on Python **3.10** & **3.12** — synthetic data only, no lattice files in CI ([workflow](.github/workflows/ci.yml)).

---

## Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

---

## Author

**Dr. Geng Li** — lattice QCD, scientific computing, HPC. For collaboration or citation, contact via repository details.

## License

Not specified. Contact the maintainer before redistribution.
