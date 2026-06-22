import numpy as np
import gvar as gv


class MathModels:
    """Pure model functions and prior factories."""

    @staticmethod
    def two_states(t, parameter, lattice_Nt: int):
        mid = lattice_Nt / 2
        return parameter["weff_0"] * gv.cosh(parameter["meff_0"] * (t - mid)) + parameter[
            "weff_1"
        ] * gv.cosh(parameter["meff_1"] * (t - mid))

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
    def ratio(T, t, p):
        return (
            p["ar"]
            * gv.sinh(p["meff_1"] * (t - T / 2))
            / gv.sinh(p["summ_mesn"] * (t - T / 2))
        )

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
            "weff_0": gv.gvar(weff_prior[0], 0.1),
            "weff_1": gv.gvar(weff_prior[1], 0.1),
            "weff_2": gv.gvar(weff_prior[2], 0.01),
            "meff_0": gv.gvar(meff_prior[0], 0.01),
            "meff_1": gv.gvar(meff_prior[1], 0.01),
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

    @staticmethod
    def prior_ratio(meff_prior):
        return {
            "ar": gv.gvar(1.0, 0.2),
            "meff_1": gv.gvar(meff_prior[0], 0.01),
            "summ_mesn": gv.gvar(0.2, 0.02),
        }


MODEL_REGISTRY = {
    "two_states": {"fn": MathModels.two_states, "prior": MathModels.prior_two_states},
    "three_states": {"fn": MathModels.three_states, "prior": MathModels.prior_three_states},
    "ratio": {"fn": MathModels.ratio, "prior": MathModels.prior_ratio},
}
