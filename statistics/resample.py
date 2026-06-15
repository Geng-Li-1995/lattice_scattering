# resample.py

import numpy as np
import gvar as gv
import lsqfit as lsf

from statistics.jackknife import Jackknife
from statistics.bootstrap import Bootstrap
from analysis.fitting import RunFitting
from analysis.models import MathModels
from analysis.gevp import process_GEVP
from input.config import Config
from input.types import FileDict


def get_resampler(config: Config, data: np.ndarray) -> Jackknife | Bootstrap:
    resample_type = getattr(config, "resample_type", "jackknife")
    axis = getattr(config, "sample_axis", -1)
    n_boot = getattr(config, "n_boot", 1000)

    if resample_type == "jackknife":
        return Jackknife(data, axis=axis)
    if resample_type == "bootstrap":
        return Bootstrap(data, axis=axis, nboot=n_boot)

    raise ValueError(f"Unknown resample_type: {resample_type}")


def run_resample_statistics(config: Config, file_dict: FileDict) -> None:
    """
    Run jackknife or bootstrap resampling and save energies to data/<system>/resampled/.
    Intended to be called from run_resample.py.
    """
    # -----------------------------
    # cache config (minor speedup)
    # -----------------------------
    sample_axis = config.sample_axis
    chan_momt_list = config.chan_momt_list
    at_invs = config.at_invs
    Ns = config.ensemble_key[0]

    input_name = config.input_name
    tag_name = config.tag_name

    is_meson = getattr(config, "is_meson_analysis", False)
    is_tetra = getattr(config, "is_tetraquark_analysis", False)

    type_name = "tetraquark" if is_tetra else "meson"

    n_chan = len(chan_momt_list)
    mom_max = max((max(m) for m in chan_momt_list if m), default=-1) + 1

    # -----------------------------
    # check sample consistency
    # -----------------------------
    n_sample = None
    for k, v in file_dict.items():
        n = v.shape[sample_axis]
        if n_sample is None:
            n_sample = n
        elif n != n_sample:
            raise ValueError("Different n_samples across keys")

    # -----------------------------
    # outputs
    # -----------------------------
    resample_type = config.resample_type
    n_iterations = config.n_boot if resample_type == "bootstrap" else n_sample

    results = np.zeros((n_chan, mom_max, n_iterations))
    ksi = np.zeros((n_chan, n_iterations)) if is_meson else None

    fitter = RunFitting(config)
    rng = np.random.default_rng()

    # -----------------------------
    # main loop
    # -----------------------------
    for sam in range(n_iterations):

        file_resample_dict = {}

        if resample_type == "jackknife":
            for key, file in file_dict.items():
                file_resample_dict[key] = Jackknife(file, axis=sample_axis).resample_manual(
                    sam
                )
        elif resample_type == "bootstrap":
            boot_idx = rng.integers(0, n_sample, size=n_sample)
            for key, file in file_dict.items():
                file_resample_dict[key] = np.take(file, boot_idx, axis=sample_axis)
        else:
            raise ValueError(f"Unknown resample_type: {resample_type}")

        data_resample_dict = process_GEVP(config, file_resample_dict)
        fit_results = fitter.effective_mass(data_resample_dict)

        # ---- build lookup ----
        fit_map = {}
        for r in fit_results:
            fit_map[(r["channel"], r["mom"])] = r["fit"]

        for ch_idx, mom_list in enumerate(chan_momt_list):

            Ens_sq = []

            for mom in mom_list:
                fit = fit_map.get((ch_idx, mom))
                if fit is None:
                    raise ValueError(f"Missing fit: channel={ch_idx}, mom={mom}")

                En = fit.p["meff_0"] * at_invs
                Ens_sq.append(En * En)

                results[ch_idx, mom, sam] = gv.mean(En)

            if is_meson:
                fit_linear = lsf.nonlinear_fit(
                    data=(np.asarray(mom_list), Ens_sq),
                    fcn=MathModels.linear,
                    prior=MathModels.prior_linear(),
                    p0=None,
                )

                ksi[ch_idx, sam] = gv.mean(
                    2 * np.pi * at_invs / np.sqrt(fit_linear.p["a"]) / Ns
                )

        print(
            "### Finished sam:",
            sam,
            "\nksi:",
            None if ksi is None else ksi[:, sam],
            "\nEn:\n",
            results[:, :, sam],
        )

    # -----------------------------
    # save
    # -----------------------------
    base = f"data/{input_name}/resampled"

    np.save(
        f"{base}/resample_En_{type_name}_{tag_name}.npy",
        results,
    )

    if is_meson:
        np.save(
            f"{base}/resample_ksi_{type_name}_{tag_name}.npy",
            ksi,
        )

    return
