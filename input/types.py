from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import numpy as np

EnsembleKey: TypeAlias = Tuple[int, int, int, int]
ScatteringList: TypeAlias = List[EnsembleKey]
CorrelatorArray: TypeAlias = np.ndarray
ResampleDataDict: TypeAlias = Dict[str, Dict[EnsembleKey, np.ndarray]]

ModelFn: TypeAlias = Callable[..., Any]
PriorFn: TypeAlias = Callable[..., Dict[str, Any]]
FitResultList: TypeAlias = List[Dict[str, Any]]
EnsembleEntry: TypeAlias = Dict[str, Any]
