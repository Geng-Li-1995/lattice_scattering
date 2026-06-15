# main.py

from input.config import BuildConfig
from data.io import read_file
from analysis.fitting import RunFitting
from statistics.resample import run_resample_statistics
from plotting.plot_mass import MassPlotter
from plotting.plot_scattering import ScatteringPlotter
from analysis.gevp import process_GEVP
from analysis.scattering import run_scattering_analysis


def main():

    # build config from input/name.py, e.g. input/Tcccc6600_input.py
    config = BuildConfig("Tcccc6600").build_config_from_control()

    # read file from data/name.npy, e.g. Tcccc6600/meson_L16M420_EV120.npy
    file_dict, resample_data_dict = read_file(config)

    # process/plot GEVP from tetraquark data
    data_dict = process_GEVP(config, file_dict)

    # fit/plot En and weights Zn/Z0 from data
    En_result_dict = RunFitting(config).effective_mass(data_dict)
    MassPlotter(config).plot_En(data_dict, En_result_dict)
    MassPlotter(config).plot_Zn(En_result_dict)

    # fit/plot dispersion relation from En
    disp_result = RunFitting(config).dispersion(En_result_dict)
    MassPlotter(config).plot_dispersion(En_result_dict, disp_result)

    # run and plot scattering analysis
    scattering_results_dict = run_scattering_analysis(config, resample_data_dict)
    ScatteringPlotter(config).plot_Ks(scattering_results_dict)
    ScatteringPlotter(config).plot_kcot(scattering_results_dict)

    # run resample statistics
    run_resample_statistics(config, file_dict)

    # finished
    print("Main task is finished!")


if __name__ == "__main__":
    main()
