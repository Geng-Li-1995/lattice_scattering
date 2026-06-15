"""Small helpers shared across analysis modules."""

import numpy as np

from input.types import FitResultList


def en_fit_lookup(en_fit_list: FitResultList) -> dict:
    return {
        (e["ch_idx"], e["mom"]): e["fit"]
        for e in en_fit_list
        if "mom" in e
    }


def disp_fit_lookup(disp_fit_list: FitResultList) -> dict:
    return {e["ch_idx"]: e["fit"] for e in disp_fit_list}


def ksi_from_disp_fit(disp_fit, at_invs: float, ns: int) -> float:
    return 2 * np.pi * at_invs / np.sqrt(disp_fit.p["a"]) / ns


def fve_offsets(n_mom_per_ch: list[int]) -> np.ndarray:
    return np.cumsum([0, *n_mom_per_ch[:-1]])
