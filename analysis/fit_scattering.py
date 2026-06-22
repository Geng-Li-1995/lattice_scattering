"""Scattering-phase fits: K(s) vs s and k cot delta vs k^2.

Fit mode is selected by ``Config.scattering_fit_mode``:

- ``Ks_linear`` (default for Tcccc6600): linear fit of K(s) = sqrt(s) / (k cot delta) in s.
- ``kcot_quadratic`` (default for X3872): quadratic fit of k cot delta in k^2.
"""

import gvar as gv
import lsqfit as lsf
import numpy as np

from analysis.models import MathModels
from input.config import Config

SCATTERING_FIT_MODES = ("Ks_linear", "kcot_quadratic")


def fit_mom_indices(config: Config, scattering_dict: dict) -> dict[int, list[int]]:
    """Return momentum-level indices used in the scattering fit."""
    if not config.is_moving_frame:
        return config.fit_mom_by_ns

    fit_mom_by_ns: dict[int, list[int]] = {}
    for ensemble_key in config.scattering_list:
        ns = ensemble_key[0]
        rest_point_count = scattering_dict["rest_point_count"][ns]
        fit_mom_by_ns[ns] = list(config.fit_mom_by_ns.get(ns, [])) + [
            rest_point_count + mom for mom in config.fit_mom_by_ns_MF.get(ns, [])
        ]
    return fit_mom_by_ns


def run_scattering_fit(
    config: Config, scattering_dict: dict, fit_mom_by_ns: dict[int, list[int]]
) -> None:
    """Run the configured fit and store curves in ``scattering_dict``."""
    if config.scattering_fit_mode not in SCATTERING_FIT_MODES:
        raise ValueError(
            f"Unknown scattering_fit_mode={config.scattering_fit_mode!r}; "
            f"expected one of {SCATTERING_FIT_MODES}"
        )

    ref_ns = next(iter(scattering_dict["s_array"]))
    if config.scattering_fit_mode == "Ks_linear":
        _fit_Ks_linear(scattering_dict, fit_mom_by_ns, ref_ns)
    else:
        _fit_kcot_quadratic(scattering_dict, fit_mom_by_ns, ref_ns)


def _concat_fit_points(
    scattering_dict: dict, fit_mom_by_ns: dict[int, list[int]], key: str
) -> np.ndarray:
    return np.concatenate(
        [scattering_dict[key][ns][moms] for ns, moms in fit_mom_by_ns.items()]
    )


def _fit_Ks_linear(
    scattering_dict: dict,
    fit_mom_by_ns: dict[int, list[int]],
    ref_ns: int,
) -> None:
    s_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, "s")
    ks_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, "Ks")
    fit_result = lsf.nonlinear_fit(
        data=(s_all[:, 0], gv.gvar(ks_all[:, 0], ks_all[:, 1])),
        fcn=MathModels.linear,
        prior=MathModels.prior_linear(),
    )
    scattering_dict["fit_result"] = fit_result
    scattering_dict["fit_Ks_curve"] = MathModels.linear(
        scattering_dict["s_array"][ref_ns], fit_result.p
    )
    scattering_dict["fit_kcot_curve"] = (
        scattering_dict["sqrt_s_array"][ref_ns] / scattering_dict["fit_Ks_curve"]
    )
    print("Ks fit:", s_all[:, 0], gv.gvar(ks_all[:, 0], ks_all[:, 1]), fit_result)


def _fit_kcot_quadratic(
    scattering_dict: dict,
    fit_mom_by_ns: dict[int, list[int]],
    ref_ns: int,
) -> None:
    k_sq_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, "k_sq")
    kcot_all = _concat_fit_points(scattering_dict, fit_mom_by_ns, "kcot")
    fit_result = lsf.nonlinear_fit(
        data=(k_sq_all[:, 0], gv.gvar(kcot_all[:, 0], kcot_all[:, 1])),
        fcn=MathModels.quadratic,
        prior=MathModels.prior_quadratic(),
    )
    scattering_dict["fit_result"] = fit_result
    scattering_dict["fit_kcot_curve"] = MathModels.quadratic(
        scattering_dict["k_sq_array"][ref_ns], fit_result.p
    )
    print(
        "kcot quadratic fit:",
        k_sq_all[:, 0],
        gv.gvar(kcot_all[:, 0], kcot_all[:, 1]),
        fit_result,
    )
