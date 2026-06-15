# plot_mass.py

import numpy as np
import gvar as gv
import lsqfit as lsf
import matplotlib.pyplot as plt

from input.selector import SelectorType
from analysis.models import MathModels
from statistics.resample import get_resampler


class MassPlotter:

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

    LINESTYLES = [
        "-",
        "--",
        ":",
        "-.",
        (0, (1, 1)),
        (0, (5, 2)),
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
        (0, (2, 2)),
        (0, (1, 2)),
    ]

    MARKERS = ["o", "x", "s", "^", "v", "<", ">", "*", "D", "p"]

    def __init__(self, config):

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
            }
        )

        self.config = config
        self.input_name = config.input_name
        self.tag_name = config.tag_name
        self.at_invs = config.at_invs
        self.lattice_Nt = config.lattice_Nt

        self.chan_momt_list = config.chan_momt_list
        self.chan_name_list = config.chan_name_list

        self.type_name = (
            "tetraquark"
            if getattr(config, "is_tetraquark_analysis", False)
            else "meson"
        )

    # ==================================================
    # helpers
    # ==================================================
    def _build_En_lookup(self, En_result):
        return {(r["channel"], r["mom"]): r["fit"] for r in En_result}

    def _cosh_resample(self, data):
        r = get_resampler(self.config, data).resample()
        c = MathModels.generate_cosh_from_data(r, time_axis=0)
        mean, err = get_resampler(self.config, c).gvar()
        return mean, err, c

    def _save(self, name):
        plt.tight_layout()
        plt.savefig(
            f"result/{self.input_name}/{name}_{self.type_name}_{self.tag_name}.pdf"
        )
        plt.show()

    # ==================================================
    # En
    # ==================================================
    def plot_En(self, data_dict, En_result):

        data = SelectorType(self.config, data_dict).get_data()
        fit_function, _ = SelectorType(self.config, data_dict).get_model()

        En_lookup = self._build_En_lookup(En_result)

        ylim_min = 1.0 if self.type_name == "tetraquark" else 0.1

        Ens = {}
        plt.figure(figsize=(13, 8))

        for ch_idx, mom_list in enumerate(self.chan_momt_list):

            Ens[ch_idx] = []

            for mom in mom_list:

                mean, err, _ = self._cosh_resample(data[ch_idx][mom])

                plt.errorbar(
                    np.arange(self.lattice_Nt),
                    mean * self.at_invs,
                    err * self.at_invs,
                    fmt=self.MARKERS[ch_idx],
                    color=self.COLORS[mom],
                    markersize=4,
                    capsize=3,
                    linewidth=1.5,
                    capthick=1,
                    markeredgecolor="black",
                    markerfacecolor="white",
                    label=f"${self.chan_name_list[ch_idx]}(n^2={mom})$",
                )

                fit = En_lookup[(ch_idx, mom)]

                tmin, tmax = (
                    self.config.t_min[ch_idx][mom],
                    self.config.t_max[ch_idx][mom],
                )
                t_fit = np.arange(tmin, tmax)

                result_fit = fit_function(t_fit, fit.p, self.lattice_Nt)
                cosh_fit = MathModels.generate_cosh_from_data(result_fit, time_axis=0)

                m = gv.mean(cosh_fit) * self.at_invs
                s = gv.sdev(cosh_fit) * self.at_invs

                plt.plot(
                    t_fit,
                    m,
                    linestyle=self.LINESTYLES[ch_idx],
                    color=self.COLORS[mom],
                )
                plt.fill_between(
                    t_fit,
                    m - s,
                    m + s,
                    color=self.COLORS[mom],
                    alpha=0.2,
                )

                Ens[ch_idx].append(fit.p["meff_0"] * self.at_invs)

        all_ens = np.concatenate([np.array(v) for v in Ens.values()])

        plt.xlabel(r"$t/a_t$", fontsize=25)
        plt.ylabel(rf"$E_n$ (GeV) on {self.tag_name}", fontsize=25)

        plt.xlim(1, self.lattice_Nt - 1)
        plt.ylim(gv.mean(np.min(all_ens)) - ylim_min, gv.mean(np.max(all_ens)) + 0.2)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.legend(loc="lower right", fontsize=20, ncol=2)

        self._save("En")

    # ==================================================
    # Zn
    # ==================================================
    def plot_Zn(self, En_result):
        if not getattr(self.config, "is_meson_analysis", False):
            return

        En_lookup = self._build_En_lookup(En_result)
        mom_max = max(max(m) for m in self.chan_momt_list if m)

        plt.figure(figsize=(8, 5))

        for ch_idx, mom_list in enumerate(self.chan_momt_list):

            fit0 = En_lookup[(ch_idx, mom_list[0])]
            Zns = []

            for mom in mom_list:
                fit = En_lookup[(ch_idx, mom)]
                Zn = fit.p["weff_0"] / fit0.p["weff_0"]
                Zns.append(Zn)

                plt.errorbar(
                    mom,
                    gv.mean(Zn),
                    gv.sdev(Zn),
                    fmt=self.MARKERS[ch_idx],
                    color=self.COLORS[ch_idx],
                    markersize=4,
                    capsize=3,
                    linewidth=1.5,
                    capthick=1,
                    markeredgecolor="black",
                    markerfacecolor="white",
                )

            fit_exp = lsf.nonlinear_fit(
                data=(np.array(mom_list), np.array(Zns)),
                fcn=MathModels.exponential,
                prior=MathModels.prior_exponential(),
            )

            x = np.linspace(0, mom_max, 100)
            plt.plot(
                x,
                gv.mean(MathModels.exponential(x, fit_exp.p)),
                linestyle=self.LINESTYLES[ch_idx],
                color=self.COLORS[ch_idx],
            )

        plt.xlabel(r"$n^2$", fontsize=20)
        plt.ylabel(rf"$Z_n/Z_0$ on {self.tag_name}", fontsize=20)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.xlim(0, mom_max)
        plt.ylim(0, 1)

        self._save("Zn")

    # ==================================================
    # dispersion
    # ==================================================
    def plot_dispersion(self, En_result, disp_result):

        if self.type_name != "meson":
            print("Plot dispersion relation only from meson En.")
            return

        En_lookup = self._build_En_lookup(En_result)
        disp_lookup = {r["channel"]: r["fit"] for r in disp_result}

        mom_max = max(max(m) for m in self.chan_momt_list if m)

        Ens_sq = {}

        plt.figure(figsize=(8, 5))

        for ch_idx, mom_list in enumerate(self.chan_momt_list):

            Ens_sq[ch_idx] = []

            for mom in mom_list:
                En = En_lookup[(ch_idx, mom)].p["meff_0"] * self.at_invs
                Ens_sq[ch_idx].append(En**2)

            disp_fit = disp_lookup[ch_idx]

            y = gv.mean(Ens_sq[ch_idx])
            yerr = gv.sdev(Ens_sq[ch_idx])

            plt.errorbar(
                mom_list,
                y,
                yerr,
                fmt=self.MARKERS[ch_idx],
                color=self.COLORS[ch_idx],
                markersize=4,
                capsize=3,
                linewidth=1.5,
                capthick=1,
                label=f"${self.chan_name_list[ch_idx]}$",
                markeredgecolor="black",
                markerfacecolor="white",
            )

            x = np.linspace(0, mom_max, 100)
            plt.plot(
                x,
                gv.mean(MathModels.linear(x, disp_fit.p)),
                linestyle=self.LINESTYLES[ch_idx],
                color=self.COLORS[ch_idx],
            )

        plt.xlabel(r"$n^2$", fontsize=20)
        plt.ylabel(rf"$E_n^2$ (GeV$^2$) on {self.tag_name}", fontsize=20)

        plt.xlim(0, mom_max)
        all_vals = np.concatenate([gv.mean(v) for v in Ens_sq.values()])
        plt.ylim(np.min(all_vals), np.max(all_vals))

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.legend(loc="upper left", fontsize=20)

        self._save("Dispersion")
