# selector.py

import numpy as np
from typing import List, Dict, Callable, Any
from analysis.models import MODEL_REGISTRY


class SelectorType:
    """
    Unified interface for:
    - Selecting data (meson / tetraquark)
    - Selecting model and prior
    """

    def __init__(self, config: Any, data: Dict[str, List[np.ndarray]]):
        self.config = config
        self.data = data

    # ======================================================
    # 0. Data selection
    # ======================================================
    def get_data(self) -> List[np.ndarray]:
        """Return the dataset according to analysis type."""

        if getattr(self.config, "is_meson_analysis", True):
            return self.data["meson"]

        if getattr(self.config, "is_tetraquark_analysis", False):
            return self.data["tetraquark"]

        raise ValueError("No valid analysis type selected in config")

    # ======================================================
    # 1. Model selection
    # ======================================================
    def get_model(self) -> tuple[Callable, Any]:
        """Return the fitting function and prior based on config."""
        if getattr(self.config, "is_ratio", False):
            function = MODEL_REGISTRY["ratio"]["fn"]
            prior = MODEL_REGISTRY["ratio"]["prior"]
            return function, prior

        n_state = getattr(self.config, "n_state", 2)
        if n_state == 2:
            function = MODEL_REGISTRY["two_states"]["fn"]
            prior = MODEL_REGISTRY["two_states"]["prior"]
            return function, prior
        elif n_state == 3:
            function = MODEL_REGISTRY["three_states"]["fn"]
            prior = MODEL_REGISTRY["three_states"]["prior"]
            return function, prior

        raise ValueError(f"Unsupported n_state={n_state}")
