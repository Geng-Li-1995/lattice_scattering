from input.config import BuildConfig
from data.io import read_file
from analysis.fitting import RunFitting
from analysis.gevp import process_GEVP
from analysis.scattering import run_scattering_analysis
from plotting.plot_gevp import GEVPPlotter
from plotting.plot_mass import MassPlotter
from plotting.plot_scattering import ScatteringPlotter


def main() -> None:
    config = BuildConfig("Tcccc6600").build_config_from_control()
    raw, resampled = read_file(config)

    corr, gevp_plot_data = process_GEVP(config, raw)
    if gevp_plot_data is not None:
        GEVPPlotter(config).plot_GEVP(
            gevp_plot_data.matrix_before,
            gevp_plot_data.matrix_after,
            gevp_plot_data.eigenvectors,
        )

    fitter = RunFitting(config)
    plotter = MassPlotter(config)

    en_fit_list = fitter.effective_mass(corr)

    if config.plot_meff:
        plotter.plot_En(corr, en_fit_list)
        plotter.plot_Zn(en_fit_list)

    if config.plot_dispersion and config.is_meson_analysis:
        disp_fit_list = fitter.dispersion(en_fit_list)
        plotter.plot_dispersion(en_fit_list, disp_fit_list)

    scattering_dict = run_scattering_analysis(config, resampled)
    if scattering_dict is not None:
        scat_plotter = ScatteringPlotter(config)
        scat_plotter.plot_Ks(scattering_dict)
        scat_plotter.plot_kcot(scattering_dict)

    print("Main task is finished!")


if __name__ == "__main__":
    main()
