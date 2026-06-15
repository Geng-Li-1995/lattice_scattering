# bootstrap.py

import numpy as np


class Bootstrap:
    """
    Bootstrap resampling and statistics for NumPy arrays.
    """

    def __init__(
        self,
        data: np.ndarray,
        axis: int = -1,
        nboot: int = 1000,
        random_seed: int | None = None,
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Input data array.
        axis : int, optional
            Sample axis along which to resample.
        nboot : int, optional
            Number of bootstrap resamples.
        random_seed : int | None, optional
            Random seed for reproducibility.
        """
        self.data = np.asarray(data)
        self.axis = axis
        self.nboot = nboot
        self.random_seed = random_seed
        self._boot_samples = None
        self._rng = np.random.default_rng(random_seed)

    @property
    def ncfg(self) -> int:
        """Number of configurations."""
        return self.data.shape[self.axis]

    def resample(self) -> np.ndarray:
        """
        Generate bootstrap resamples with replacement.

        Returns
        -------
        np.ndarray
            Bootstrap samples with shape (nboot, ...) where ... is the remaining shape of data.
        """
        if self._boot_samples is None:
            n = self.ncfg
            data = np.moveaxis(self.data, self.axis, 0)
            shape = (self.nboot, *data.shape[1:])

            # Sample indices with replacement
            indices = self._rng.integers(0, n, size=(self.nboot, n))
            self._boot_samples = data[indices]

            # Move the sample axis back if necessary
            self._boot_samples = np.moveaxis(self._boot_samples, 1, self.axis + 1)

        return self._boot_samples

    def mean(self) -> np.ndarray:
        """
        Mean of the original data.

        Returns
        -------
        np.ndarray
        """
        return np.mean(self.data, axis=self.axis)

    def error(self) -> np.ndarray:
        """
        Compute bootstrap standard error from resampled data.

        Returns
        -------
        np.ndarray
            Bootstrap standard error.
        """
        boot_samples = self.resample()
        return np.std(boot_samples, axis=self.axis, ddof=0)

    def gvar(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return mean and bootstrap error of the original data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
        """
        return self.mean(), self.error()

    def resample_manual(self, idx: np.ndarray | None = None) -> np.ndarray:
        """
        Generate a bootstrap resample by manually specifying indices.

        Parameters
        ----------
        idx : np.ndarray | None
            Array of indices to sample along the sample axis. If None, generates one random resample.

        Returns
        -------
        np.ndarray
            Array with bootstrap resampling applied along the sample axis.
        """
        n = self.ncfg
        if idx is None:
            idx = self._rng.integers(0, n, size=n)
        elif not isinstance(idx, np.ndarray):
            idx = np.asarray(idx)

        return np.take(self.data, idx, axis=self.axis)
