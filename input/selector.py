from input.config import Config
from input.types import CorrelatorArray, FileDict, ModelFn, PriorFn
from analysis.models import MODEL_REGISTRY


class SelectorType:
    """Select correlator data and fitting model from config."""

    def __init__(self, config: Config, corr_dict: FileDict):
        self.config = config
        self.corr_dict = corr_dict

    def get_data(self) -> CorrelatorArray:
        if self.config.is_meson_analysis:
            return self.corr_dict["meson"]
        if self.config.is_tetraquark_analysis:
            return self.corr_dict["tetraquark"]
        raise ValueError("No valid analysis type selected in config")

    def get_model(self) -> tuple[ModelFn, PriorFn]:
        if self.config.is_ratio:
            return MODEL_REGISTRY["ratio"]["fn"], MODEL_REGISTRY["ratio"]["prior"]

        key = {2: "two_states", 3: "three_states"}.get(self.config.n_state)
        if key is None:
            raise ValueError(f"Unsupported n_state={self.config.n_state}")
        return MODEL_REGISTRY[key]["fn"], MODEL_REGISTRY[key]["prior"]
