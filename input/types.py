# types.py
#
# Naming conventions (local variables):
#   raw_dict       – raw correlators loaded from disk
#   corr_dict      – correlators after GEVP, ready for fitting
#   resampled_dict – jackknife/bootstrap energies and ksi
#   en_fit_list    – effective-mass fit results
#   disp_fit_list  – dispersion-relation fit results
#   scattering_dict – scattering analysis outputs
#   ch_idx         – channel index
#   mom            – momentum quantum number n^2 (array index)
#   mom_list       – momentum values for one channel
#   ensemble_key   – (Ns, Nt, pion_mass, num_eigenvectors)
#   corr_type      – "meson" or "tetraquark"
#   sample_idx     – jackknife/bootstrap iteration index
#   matrix_before_gevp / matrix_after_gevp – FVE matrices before/after GEVP
#   ch_meson_a/b, ch_tetra – meson/tetraquark channel indices for scattering
#   fit_mom_by_ns          – momentum subsets for K(s) fit, keyed by Ns

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import numpy as np

EnsembleKey: TypeAlias = Tuple[int, int, int, int]
ScatteringList: TypeAlias = List[EnsembleKey]
CorrelatorArray: TypeAlias = np.ndarray
FileDict: TypeAlias = Dict[str, CorrelatorArray]
ResampleDataDict: TypeAlias = Dict[str, Dict[EnsembleKey, np.ndarray]]

ModelFn: TypeAlias = Callable[..., Any]
PriorFn: TypeAlias = Callable[..., Dict[str, Any]]

# Fit result entry: {"ch_idx", "mom", "fit"} or {"ch_idx", "fit"} for dispersion
FitEntry: TypeAlias = Dict[str, Any]
FitResultList: TypeAlias = List[FitEntry]

EnsembleEntry: TypeAlias = Dict[str, Any]
