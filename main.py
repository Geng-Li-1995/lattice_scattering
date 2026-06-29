from input.config import BuildConfig, Config
from data.io import read_raw_files, read_resampled_files
from analysis.fit_mass import RunFitting
from analysis.gevp import process_GEVP
from analysis.scattering import run_scattering_analysis
from plotting.plot_gevp import GEVPPlotter
from plotting.plot_mass import MassPlotter
from plotting.plot_scattering import ScatteringPlotter
from statistics.resample import run_resample_statistics


def run_analysis(
    config: Config,
    meson_config: Config | None = None,
) -> None:
    if not (config.run_meson_analysis or config.run_tetraquark_analysis):
        return

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

    plotter.plot_En(corr, en_fit_list)
    if config.run_weight_analysis:
        plotter.plot_Zn(en_fit_list)
    if config.run_ratio_analysis:
        plotter.plot_ratio_workflow(corr, meson_config)

    if config.run_dispersion_analysis:
        disp_fit_list = fitter.dispersion(en_fit_list)
        plotter.plot_dispersion(en_fit_list, disp_fit_list)

    if config.run_tmin_analysis:
        plotter.plot_tmin_workflow(corr, meson_config)


def run_resample_analysis(config: Config) -> None:
    if not (config.run_meson_analysis or config.run_tetraquark_analysis):
        return
    raw = read_raw_files(config)
    run_resample_statistics(config, raw)


def _run_analysis_branch(
    builder: BuildConfig, analysis_type: str, ctrl, meson_config: Config | None = None
) -> None:
    enabled = (
        ctrl.run_meson_analysis
        if analysis_type == "meson"
        else ctrl.run_tetraquark_analysis
    )
    if not enabled:
        return

    config = builder.build_config_from_control(analysis_type)
    if ctrl.run_resample_analysis:
        run_resample_analysis(config)
    else:
        run_analysis(
            config,
            meson_config if analysis_type == "tetraquark" else None,
        )


def main() -> None:
    builder = BuildConfig("Tcccc6600")
    ctrl = builder.input_control

    # Meson Config for tetra ratio / t_min (meson priors & windows); raw meson loaded when run_ratio_analysis.
    meson_config = (
        builder.build_config_from_control("meson")
        if ctrl.run_tetraquark_analysis and ctrl.run_ratio_analysis
        else None
    )

    _run_analysis_branch(builder, "meson", ctrl)
    _run_analysis_branch(builder, "tetraquark", ctrl, meson_config)

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
