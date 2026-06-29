from dataclasses import replace

import pytest

from analysis.scattering import run_scattering_analysis
from input.config import BuildConfig


def test_tetra_config_clears_meson_only_weight_and_dispersion():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(
        builder.input_control,
        run_dispersion_analysis=True,
        run_weight_analysis=True,
    )
    config = builder.build_config_from_control("tetraquark")
    assert config.run_dispersion_analysis is False
    assert config.run_weight_analysis is False


def test_meson_config_clears_tetra_only_gevp():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(builder.input_control, run_GEVP_analysis=True)
    config = builder.build_config_from_control("meson")
    assert config.run_GEVP_analysis is False


def test_is_plot_title_passed_to_config():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(builder.input_control, is_plot_title=False)
    config = builder.build_config_from_control("tetraquark")
    assert config.is_plot_title is False


def test_is_plot_show_passed_to_config():
    builder = BuildConfig("Tcccc6600")
    builder.input_control = replace(builder.input_control, is_plot_show=False)
    config = builder.build_config_from_control("tetraquark")
    assert config.is_plot_show is False


def test_scattering_not_gated_by_run_tetraquark():
    config = BuildConfig("Tcccc6600").build_config_from_control("tetraquark")
    config = replace(config, run_tetraquark_analysis=False, run_scattering_analysis=True)
    with pytest.raises(KeyError):
        run_scattering_analysis(config, {})
