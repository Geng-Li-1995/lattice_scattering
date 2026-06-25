# Lattice QCD Tetraquark Scattering

[![CI](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/Geng-Li-1995/lattice_scattering/actions/workflows/ci.yml)

Config-driven analysis for lattice tetraquark spectroscopy and finite-volume scattering: correlator I/O, GEVP, Bayesian multi-state fits, jackknife/bootstrap resampling, and Lüscher extraction.

Switch systems via `BuildConfig("<System>")` in `main.py`. Physics choices live in `input/input_<System>.py`.

**Reference system:** Tcccc6600 — fully-charm \(\eta_c\eta_c\) / \(J/\psi\,J/\psi\) on \(L=12,16\), \(N_t=96/128\), \(m_\pi=420\) MeV, \(a_t^{-1}=7.219\) GeV, 400 gauge configs per volume.

Raw and resampled `.npy` data are not in git; place them under `data/<system>/` locally.

---

## Systems

| System | Input | Ensembles | Channels |
|--------|-------|-----------|----------|
| `Tcccc6600` | `input/input_Tcccc6600.py` | \(L=12,16\), EV 170/120 | \(\eta_c\eta_c\), \(J/\psi J/\psi\) |
| `X3872` | `input/input_X3872.py` | \(L=16\), EV 70 | \(\pi J/\psi\), \(\rho\eta_c\), \(DD^*\), \(D^*D^*\) |
| `Zc3900` | `input/input_Zc3900.py` | \(L=16\), EV 70 | same as X3872 |

Array shapes: meson `[channel, mom, time, sample]`; tetraquark raw `[ch_src, mom_src, ch_snk, mom_snk, time, sample]`; after GEVP tetraquark is 4D like meson. Here `mom` is \(n^2\) used as an array index.

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

Setup details: [docs/RUNNING.md](docs/RUNNING.md). Tests: [docs/TESTING.md](docs/TESTING.md). Dependencies: [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

---

## Workflow

```
data/<system>/raw/*.npy
       │
       ▼ main.py  (InputControl switches)
  run_resample_statistics()     → data/<system>/resampled/*.npy
  process_GEVP()                → 4D tetraquark correlators
  effective_mass()              → En fits
  plot_tmin_workflow()          → optional t_min scan + ratio
  run_scattering_analysis()     → K(s), k cot δ₀
       │
       ▼
  result/<system>/*.{png,pdf}
```

`run_resample=True` runs leave-one-out jackknife over all gauge configs (expensive). Scattering reads precomputed `resampled/` files.

---

## Repository layout

```
main.py
input/          config.py, input_<System>.py (ENSEMBLE_DB)
data/           correlators.py, io.py
analysis/       gevp.py, fit_mass.py, fit_tmin.py, scattering.py, models.py
statistics/     jackknife.py, bootstrap.py, resample.py
plotting/       plot_set.py, plot_gevp.py, plot_mass.py, plot_tmin.py, plot_scattering.py
tests/
docs/
```

---

## Output figures

Extension from `plot_format` (`png` or `pdf`). Per-ensemble tag: `L{Ns}M{M}_EV{EV}` (`EV` = distillation eigenvectors, not gauge configs).

| Plot | Pattern |
|------|---------|
| GEVP | `GEVP_{before,after,eigenvector}_{tag}.{ext}` |
| Effective mass | `En_{meson,tetraquark}_{tag}.{ext}` |
| Overlap / dispersion | `Zn_meson_*`, `Dispersion_meson_*` (meson mode) |
| \(t_{\min}\) scan | `E{n}_mom{n2}_tmin_{tag}.{ext}` or `*_tmin_ratio_*` |
| Scattering | `K_s_scattering.{ext}`, `kcot_scattering.{ext}` (all `Ns_list` volumes) |

`E1` / `E2` = first / second tetraquark channel in `chan_momt_list`. `mom{k}` = \(n^2=k\). For Tcccc6600 ratio plots, meson \(C_2\) uses the same `(channel, n^2)` as the GEVP diagonal unless `ratio_scan_points` gives a 4-tuple override.

---

## Example results (Tcccc6600)

GEVP and effective mass (\(L=12\)):

<p align="center">
  <img src="result/Tcccc6600/GEVP_before_L12M420_EV170.png" width="48%" />
  <img src="result/Tcccc6600/GEVP_after_L12M420_EV170.png" width="48%" />
</p>

<p align="center">
  <img src="result/Tcccc6600/En_tetraquark_L12M420_EV170.png" width="48%" />
  <img src="result/Tcccc6600/En_meson_L12M420_EV170.png" width="48%" />
</p>

Scattering (\(L=12+16\)):

<p align="center">
  <img src="result/Tcccc6600/K_s_scattering.png" width="48%" />
  <img src="result/Tcccc6600/kcot_scattering.png" width="48%" />
</p>

\(t_{\min}\) scan with 4Q/2Q ratio — \(J/\psi\,J/\psi\) at \(n^2=0,1\) (\(L=16\); 3-state ○, ratio ×, Combine band):

<p align="center">
  <img src="result/Tcccc6600/E2_mom0_tmin_ratio_L16M420_EV120.png" width="48%" />
  <img src="result/Tcccc6600/E2_mom1_tmin_ratio_L16M420_EV120.png" width="48%" />
</p>

---

## Configuration

Edit `input/input_<System>.py`. Common switches:

```python
lattice_Ns: int = 12
is_meson_analysis: bool = True
is_tetraquark_analysis: bool = True
is_gevp: bool = True
run_tmin: bool = True
is_ratio: bool = True
run_resample: bool = False
run_scattering: bool = True
plot_meff: bool = True
plot_format: str = "png"
resample_type: str = "jackknife"
```

Meson and tetraquark branches are independent. Both off → skip raw analysis; scattering can still run from existing `resampled/`.

---

## Tests

GitHub Actions runs `pytest` on Python 3.10 and 3.12 without lattice data. See [docs/TESTING.md](docs/TESTING.md).

---

## Publications

- G. Li, C. Shi, Y. Chen, and W. Sun, [*Scalar and Tensor Structures in $J/\psi J/\psi$ Scattering from Lattice QCD*](https://arxiv.org/abs/2505.24213), arXiv:2505.24213 [hep-lat]
- G. Li, C. Shi, Y. Chen, and W. Sun, [*$\eta_c\eta_c$ and $J/\psi J/\psi$ scattering from lattice QCD*](https://arxiv.org/abs/2505.23220), arXiv:2505.23220 [hep-lat]

---

## License

Not specified. Contact the maintainer before redistribution.
