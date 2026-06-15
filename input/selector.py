from input.config import Config
from data.correlators import AnalysisCorrelators, Correlator4D
from input.types import ModelFn, PriorFn
from analysis.models import MODEL_REGISTRY


class SelectorType:
    """Select correlator data and fitting model from config."""

    def __init__(self, config: Config, corr: AnalysisCorrelators):
        self.config = config
        self.corr = corr

    def get_data(self) -> Correlator4D:
        return self.corr.active(
            self.config.is_meson_analysis,
            self.config.is_tetraquark_analysis,
        )

    def get_model(self) -> tuple[ModelFn, PriorFn]:
        if self.config.is_ratio:
            return MODEL_REGISTRY["ratio"]["fn"], MODEL_REGISTRY["ratio"]["prior"]

        key = {2: "two_states", 3: "three_states"}.get(self.config.n_state)
        if key is None:
            raise ValueError(f"Unsupported n_state={self.config.n_state}")
        return MODEL_REGISTRY[key]["fn"], MODEL_REGISTRY[key]["prior"]
