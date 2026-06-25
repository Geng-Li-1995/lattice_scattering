from input.config import BuildConfig
from plotting.plot_tmin import TMIN_YLIM_ERR_SCALE, TminPlotter


def test_tmin_y_limits_scale_with_combine_error():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    plotter = TminPlotter(config)
    center, err = 6.09, 0.002
    lo, hi = plotter._y_limits(center, err)
    half = err * TMIN_YLIM_ERR_SCALE
    assert lo == center - half
    assert hi == center + half


def test_tmin_plotter_reads_gev_scale_from_config():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    plotter = TminPlotter(config)
    assert plotter.at_invs == config.at_invs
