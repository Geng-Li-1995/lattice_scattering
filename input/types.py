# types.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import numpy as np

# Lattice ensemble key: (Ns, Nt, pion_mass, num_eigenvectors)
EnsembleKey: TypeAlias = Tuple[int, int, int, int]

# List of ensembles used in scattering analysis (e.g. L12 and L16)
ScatteringList: TypeAlias = List[EnsembleKey]

# Correlator array indexed as [channel, momentum, time, sample]
CorrelatorArray: TypeAlias = np.ndarray

# Raw / processed correlators keyed by type name ("meson", "tetraquark")
FileDict: TypeAlias = Dict[str, CorrelatorArray]

# Resampled energies / ksi keyed by type, then by ensemble
ResampleDataDict: TypeAlias = Dict[str, Dict[EnsembleKey, np.ndarray]]

# Fitting model and prior factory returned by SelectorType.get_model()
ModelFn: TypeAlias = Callable[..., Any]
PriorFn: TypeAlias = Callable[..., Dict[str, Any]]

# Single effective-mass fit entry
FitEntry: TypeAlias = Dict[str, Any]
FitResultList: TypeAlias = List[FitEntry]

# ENSEMBLE_DB entry for one ensemble
EnsembleEntry: TypeAlias = Dict[str, Any]
