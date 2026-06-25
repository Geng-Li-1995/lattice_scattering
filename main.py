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
    *,
    skip_tmin: bool = False,
) -> None:
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

    if config.run_tmin and config.plot_tmin and not skip_tmin:
        plotter.plot_tmin_workflow(corr, meson_config)


def run_resample(config: Config) -> None:
    raw = read_raw_files(config)
    run_resample_statistics(config, raw)


def _run_analysis_branch(
    builder: BuildConfig, analysis_type: str, ctrl, meson_config: Config | None = None
) -> None:
    enabled = (
        ctrl.is_meson_analysis
        if analysis_type == "meson"
        else ctrl.is_tetraquark_analysis
    )
    if not enabled:
        return

    config = builder.build_config_from_control(analysis_type)
    skip_tmin = (
        analysis_type == "meson"
        and ctrl.is_ratio
        and ctrl.is_tetraquark_analysis
        and ctrl.run_tmin
    )
    if ctrl.run_resample:
        run_resample(config)
    else:
        run_analysis(
            config,
            meson_config if analysis_type == "tetraquark" else None,
            skip_tmin=skip_tmin,
        )


def main() -> None:
    builder = BuildConfig("Tcccc6600")
    ctrl = builder.input_control

    meson_config = (
        builder.build_config_from_control("meson")
        if ctrl.is_tetraquark_analysis
        and ctrl.run_tmin
        and (ctrl.is_ratio or ctrl.ratio_scan_points)
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
