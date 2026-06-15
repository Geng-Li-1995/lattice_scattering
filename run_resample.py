# run_resample.py
"""
Jackknife / bootstrap resampling — run separately before scattering analysis.

Generates per-configuration energies (and meson xi) under data/<system>/resampled/.
"""

from dataclasses import replace

from input.config import BuildConfig
from data.io import read_raw_files
from statistics.resample import run_resample_statistics


def main(system_name: str = "Tcccc6600") -> None:
    config = BuildConfig(system_name).build_config_from_control()
    # Suppress GEVP / fit plots and verbose output during the resample loop
    config = replace(config, run_resample=True)

    file_dict = read_raw_files(config)
    run_resample_statistics(config, file_dict)

    print("Resample task is finished!")


if __name__ == "__main__":
    main()
