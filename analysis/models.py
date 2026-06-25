import numpy as np
import gvar as gv


class MathModels:
    """Pure model functions and prior factories."""

    @staticmethod
    def two_states(t, parameter, lattice_Nt: int):
        mid = lattice_Nt / 2
        return parameter["weff_0"] * gv.cosh(
            parameter["meff_0"] * (t - mid)
        ) + parameter["weff_1"] * gv.cosh(parameter["meff_1"] * (t - mid))

    @staticmethod
    def three_states(t, parameter, lattice_Nt: int):
        mid = lattice_Nt / 2
        return (
            parameter["weff_0"] * gv.cosh(parameter["meff_0"] * (t - mid))
            + parameter["weff_1"]
            + parameter["weff_2"] * gv.cosh(parameter["meff_2"] * (t - mid))
        )

    @staticmethod
    def exponential(x, parameter):
        return np.exp(-parameter["Ek"] * x)

    @staticmethod
    def linear(x, parameter):
        return parameter["a"] * x + parameter["b"]

    @staticmethod
    def quadratic(x, parameter):
        return parameter["a"] * x**2 + parameter["b"] * x + parameter["c"]

    @staticmethod
    def ratio(t, parameter, lattice_Nt: int, meson_m: float):
        mid = lattice_Nt / 2
        return (
            parameter["ar"]
            * gv.sinh(parameter["delta_m"] * (t - mid))
            / gv.sinh(2 * meson_m * (t - mid))
        )

    @staticmethod
    def prior_ratio_delta(preset: float, ar_mean: float = 5.0, ar_sdev: float = 5.0):
        return {
            "ar": gv.gvar(ar_mean, ar_sdev),
            "delta_m": gv.gvar(preset, 0.01),
        }

    @staticmethod
    def generate_cosh_from_data(data: np.ndarray, time_axis: int = 0) -> np.ndarray:
        return np.arccosh(
            (np.roll(data, -1, axis=time_axis) + np.roll(data, 1, axis=time_axis))
            / (2 * data)
        )

    @staticmethod
    def prior_two_states(meff_prior, weff_prior):
        return {
            "weff_0": gv.gvar(weff_prior[0], 0.1),
            "weff_1": gv.gvar(weff_prior[1], 0.01),
            "meff_0": gv.gvar(meff_prior[0], 0.01),
            "meff_1": gv.gvar(meff_prior[1], 0.1),
        }

    @staticmethod
    def prior_three_states(meff_prior, weff_prior):
        return {
            "weff_0": gv.gvar(weff_prior[0], 1.0),
            "weff_1": gv.gvar(weff_prior[1], 1.0),
            "weff_2": gv.gvar(weff_prior[2], 0.01),
            "meff_0": gv.gvar(meff_prior[0], 0.01),
            "meff_2": gv.gvar(meff_prior[2], 0.1),
        }

    @staticmethod
    def prior_exponential():
        return {"Ek": gv.gvar(0, 10)}

    @staticmethod
    def prior_linear():
        return {"a": gv.gvar(0, 100), "b": gv.gvar(0, 100)}

    @staticmethod
    def prior_quadratic():
        return {"a": gv.gvar(0, 100), "b": gv.gvar(0, 100), "c": gv.gvar(0, 100)}

MODEL_REGISTRY = {
    "two_states": {"fn": MathModels.two_states, "prior": MathModels.prior_two_states},
    "three_states": {
        "fn": MathModels.three_states,
        "prior": MathModels.prior_three_states,
    },
}
