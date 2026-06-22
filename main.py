from input.config import BuildConfig, Config
from data.io import read_raw_files, read_resampled_files
from analysis.fitting import RunFitting
from analysis.gevp import process_GEVP
from analysis.scattering import run_scattering_analysis
from plotting.plot_gevp import GEVPPlotter
from plotting.plot_mass import MassPlotter
from plotting.plot_scattering import ScatteringPlotter
from statistics.resample import run_resample_statistics


def run_analysis(config: Config) -> None:
    raw = read_raw_files(config)
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
        if config.is_meson_analysis:
            plotter.plot_Zn(en_fit_list)

    if config.plot_dispersion and config.is_meson_analysis:
        disp_fit_list = fitter.dispersion(en_fit_list)
        plotter.plot_dispersion(en_fit_list, disp_fit_list)


def run_resample(config: Config) -> None:
    raw = read_raw_files(config)
    run_resample_statistics(config, raw)


def main() -> None:
    builder = BuildConfig("X3872")
    ctrl = builder.input_control

    if ctrl.is_meson_analysis:
        meson_config = builder.build_config_from_control("meson")
        if ctrl.run_resample:
            run_resample(meson_config)
        else:
            run_analysis(meson_config)

    if ctrl.is_tetraquark_analysis:
        tetraquark_config = builder.build_config_from_control("tetraquark")
        if ctrl.run_resample:
            run_resample(tetraquark_config)
        else:
            run_analysis(tetraquark_config)

    scattering_config = builder.build_config_from_control("tetraquark")
    resampled = read_resampled_files(scattering_config)
    scattering_dict = run_scattering_analysis(scattering_config, resampled)
    if scattering_dict is not None:
        scat_plotter = ScatteringPlotter(scattering_config)
        scat_plotter.plot_Ks(scattering_dict)
        scat_plotter.plot_kcot(scattering_dict)

    print("Main task is finished!")


if __name__ == "__main__":
    main()
