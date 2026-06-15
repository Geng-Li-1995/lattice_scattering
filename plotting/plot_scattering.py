# plot_scattering.py

import numpy as np
import gvar as gv
import matplotlib.pyplot as plt


class ScatteringPlotter:

    COLORS = [
        "blue",
        "red",
        "green",
        "violet",
        "orange",
        "black",
        "cyan",
        "navy",
        "yellow",
        "brown",
    ]

    def __init__(self, config):
        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name

    def _save(self, name):
        plt.tight_layout()
        plt.savefig(f"result/{self.input_name}/{name}_scattering_{self.tag_name}.pdf")
        plt.show()

    # ==================================================
    # Ks
    # ==================================================
    def plot_Ks(self, results_dict):

        if not getattr(self.config, "is_tetraquark_analysis", False):
            return

        plt.figure(figsize=(8, 6))

        for idx, scattering in enumerate(self.config.scattering_list):
            Ns = scattering[0]

            plt.errorbar(
                results_dict["s"][Ns][:, 0],
                results_dict["Ks"][Ns][:, 0],
                xerr=results_dict["s"][Ns][:, 1],
                yerr=results_dict["Ks"][Ns][:, 1],
                fmt="x",
                color=self.COLORS[idx],
                markersize=4,
                capsize=3,
                linewidth=1.5,
                capthick=1,
                label=f"L{Ns}",
                markeredgecolor="black",
                markerfacecolor="white",
            )

            if idx == 0:
                plt.plot(
                    results_dict["s_array"][Ns],
                    gv.mean(results_dict["fit_Ks_curve"]),
                    color="green",
                    linestyle="-",
                )

                plt.fill_between(
                    results_dict["s_array"][Ns],
                    gv.mean(results_dict["fit_Ks_curve"])
                    - gv.sdev(results_dict["fit_Ks_curve"]),
                    gv.mean(results_dict["fit_Ks_curve"])
                    + gv.sdev(results_dict["fit_Ks_curve"]),
                    color="green",
                    alpha=0.2,
                )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")

        plt.xlim(37, 45)
        plt.ylim(-9, 6)

        plt.xlabel(r"$s\,(\rm{GeV}^2)$", fontsize=20)
        plt.ylabel(r"$K(s)=\sqrt s / (k\cot\delta_0)$", fontsize=20)

        plt.legend(loc="upper left", fontsize=20)

        self._save("K_s")

    # ==================================================
    # kcot
    # ==================================================
    def plot_kcot(self, results_dict):
        if not getattr(self.config, "is_tetraquark_analysis", False):
            return

        plt.figure(figsize=(8, 6))

        for idx, scattering in enumerate(self.config.scattering_list):
            Ns = scattering[0]

            k_sq_array = results_dict["k_sq_array"][Ns]
            kcot_rest_array = results_dict["kcot_rest_array"][Ns]

            plt.plot(
                k_sq_array,
                kcot_rest_array,
                color=self.COLORS[idx],
                linestyle="--",
                alpha=0.3,
            )

            k_sq_mean = results_dict["k_sq"][Ns][:, 0]
            k_sq_error = results_dict["k_sq"][Ns][:, 1]
            kcot_mean = results_dict["kcot"][Ns][:, 0]
            kcot_error = results_dict["kcot"][Ns][:, 1]

            mask = np.zeros_like(k_sq_array, dtype=bool)

            for m, e in zip(k_sq_mean, k_sq_error):
                mask = (k_sq_array >= m - e) & (k_sq_array <= m + e)
                plt.plot(
                    k_sq_array[mask],
                    kcot_rest_array[mask],
                    color=self.COLORS[idx],
                    linestyle="-",
                )

            plt.errorbar(
                k_sq_mean,
                kcot_mean,
                xerr=k_sq_error,
                fmt="x",
                color=self.COLORS[idx],
                markersize=4,
                capsize=3,
                linewidth=1.5,
                capthick=1,
                label=f"L{Ns}",
                markeredgecolor="black",
                markerfacecolor="white",
            )

            if idx == 0:
                plt.plot(
                    k_sq_array,
                    gv.mean(results_dict["fit_kcot_curve"]),
                    color="green",
                    linestyle="-",
                )

                plt.fill_between(
                    k_sq_array,
                    gv.mean(results_dict["fit_kcot_curve"])
                    - gv.sdev(results_dict["fit_kcot_curve"]),
                    gv.mean(results_dict["fit_kcot_curve"])
                    + gv.sdev(results_dict["fit_kcot_curve"]),
                    color="green",
                    alpha=0.2,
                )

        plt.axhline(0, color="black")
        plt.axvline(0, color="black")

        plt.xlim(-0.25, 1.75)
        plt.ylim(-15, 15)

        plt.xlabel(r"$k^2\,(\rm{GeV}^2)$", fontsize=20)
        plt.ylabel(r"$k\cot\delta_0\,$(GeV)", fontsize=20)

        plt.legend(loc="upper left", fontsize=20)

        self._save("kcot")
