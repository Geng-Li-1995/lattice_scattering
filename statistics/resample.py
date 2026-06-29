import numpy as np
import gvar as gv

from analysis.fit_mass import RunFitting, en_fit_lookup, fit_dispersion, ksi_from_disp_fit
from analysis.gevp import process_GEVP
from data.io import save_resample_en, save_resample_ksi_meson
from data.correlators import RawCorrelators
from input.config import Config
from statistics.bootstrap import Bootstrap
from statistics.jackknife import Jackknife


def get_resampler(config: Config, data: np.ndarray) -> Jackknife | Bootstrap:
    if config.resample_type == "jackknife":
        return Jackknife(data, axis=config.sample_axis)
    if config.resample_type == "bootstrap":
        return Bootstrap(data, axis=config.sample_axis, nboot=config.n_boot)
    raise ValueError(f"Unknown resample_type: {config.resample_type}")


def jackknife_gvar(config: Config, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and jackknife error along ``sample_axis`` (same as ``plot_En`` data points)."""
    return get_resampler(config, data).gvar()


def jackknife_map_gvar(
    config: Config, data: np.ndarray, transform
) -> tuple[np.ndarray, np.ndarray]:
    """Resample → ``transform(jackknife_samples)`` → jackknife mean/error."""
    mapped = transform(get_resampler(config, data).resample())
    return get_resampler(config, mapped).gvar()


def _resample_arrays(
    config: Config,
    arrays: dict[str, np.ndarray],
    sample_idx: int,
    n_sample: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    sample_axis = config.sample_axis

    if config.resample_type == "jackknife":
        return {
            key: Jackknife(arr, axis=sample_axis).resample_manual(sample_idx)
            for key, arr in arrays.items()
        }

    if config.resample_type == "bootstrap":
        boot_idx = rng.integers(0, n_sample, size=n_sample)
        return {key: np.take(arr, boot_idx, axis=sample_axis) for key, arr in arrays.items()}

    raise ValueError(f"Unknown resample_type: {config.resample_type}")


def run_resample_statistics(config: Config, raw: RawCorrelators) -> None:
    sample_axis = config.sample_axis
    chan_momentum_list = config.chan_momentum_list
    at_invs = config.at_invs
    Ns = config.ensemble_key[0]
    analysis_corr_type = "tetraquark" if config.run_tetraquark_analysis else "meson"

    arrays = raw.array_items()
    n_chan = len(chan_momentum_list)
    mom_max = max((max(moms) for moms in chan_momentum_list if moms), default=-1) + 1

    n_samples = {arr.shape[sample_axis] for arr in arrays.values()}
    if len(n_samples) != 1:
        raise ValueError("Different n_samples across correlator types")
    n_sample = n_samples.pop()

    n_iterations = config.n_boot if config.resample_type == "bootstrap" else n_sample
    en_samples = np.zeros((n_chan, mom_max, n_iterations))
    ksi_samples = np.zeros((n_chan, n_iterations)) if config.run_meson_analysis else None

    fitter = RunFitting(config)
    rng = np.random.default_rng()

    for sample_idx in range(n_iterations):
        resampled_arrays = _resample_arrays(config, arrays, sample_idx, n_sample, rng)
        raw_jk = RawCorrelators.from_parts(**resampled_arrays)
        corr_jk, _ = process_GEVP(config, raw_jk)
        fit_lookup = en_fit_lookup(fitter.effective_mass(corr_jk))

        for chan_idx, mom_list in enumerate(chan_momentum_list):
            en_sq_list = []
            for mom in mom_list:
                mass_fit = fit_lookup.get((chan_idx, mom))
                if mass_fit is None:
                    raise ValueError(f"Missing fit: chan_idx={chan_idx}, mom={mom}")

                en = mass_fit.p["meff_0"] * at_invs
                en_sq_list.append(en * en)
                en_samples[chan_idx, mom, sample_idx] = gv.mean(en)

            if config.run_meson_analysis:
                disp_fit = fit_dispersion(mom_list, en_sq_list)
                ksi_samples[chan_idx, sample_idx] = gv.mean(
                    ksi_from_disp_fit(disp_fit, at_invs, Ns)
                )

        print(
            f"### Finished sample_idx: {sample_idx}",
            f"\nksi: {None if ksi_samples is None else ksi_samples[:, sample_idx]}",
            f"\nEn:\n{en_samples[:, :, sample_idx]}",
        )

    en_path = save_resample_en(
        config, analysis_corr_type, config.ensemble_key, en_samples
    )
    print(f"Wrote {en_path}")
    if config.run_meson_analysis:
        ksi_path = save_resample_ksi_meson(config, config.ensemble_key, ksi_samples)
        print(f"Wrote {ksi_path}")
