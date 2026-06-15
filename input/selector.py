# selector.py

from input.config import Config
from input.types import CorrelatorArray, FileDict, ModelFn, PriorFn
from analysis.models import MODEL_REGISTRY


class SelectorType:
    """
    Unified interface for:
    - Selecting data (meson / tetraquark)
    - Selecting model and prior
    """

    def __init__(self, config: Config, data: FileDict):
        self.config = config
        self.data = data

    # ======================================================
    # 0. Data selection
    # ======================================================
    def get_data(self) -> CorrelatorArray:
        """Return correlator array [channel, momentum, time, sample]."""

        if self.config.is_meson_analysis:
            return self.data["meson"]

        if self.config.is_tetraquark_analysis:
            return self.data["tetraquark"]

        raise ValueError("No valid analysis type selected in config")

    # ======================================================
    # 1. Model selection
    # ======================================================
    def get_model(self) -> tuple[ModelFn, PriorFn]:
        """Return the fitting function and prior based on config."""
        if self.config.is_ratio:
            function = MODEL_REGISTRY["ratio"]["fn"]
            prior = MODEL_REGISTRY["ratio"]["prior"]
            return function, prior

        n_state = self.config.n_state
        if n_state == 2:
            function = MODEL_REGISTRY["two_states"]["fn"]
            prior = MODEL_REGISTRY["two_states"]["prior"]
            return function, prior
        elif n_state == 3:
            function = MODEL_REGISTRY["three_states"]["fn"]
            prior = MODEL_REGISTRY["three_states"]["prior"]
            return function, prior

        raise ValueError(f"Unsupported n_state={n_state}")
