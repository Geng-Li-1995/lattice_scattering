# jackknife.py

import numpy as np


class Jackknife:
    """
    Jackknife resampling and statistics for NumPy arrays.
    """

    def __init__(self, data: np.ndarray, axis: int = -1):
        """
        Parameters
        ----------
        data : np.ndarray
            Input data array.
        axis : int, optional
            Configuration axis.
        """
        self.data = np.asarray(data)
        self.axis = axis
        self._jack_samples = None

    @property
    def ncfg(self) -> int:
        """Number of configurations."""
        return self.data.shape[self.axis]

    def resample(self) -> np.ndarray:
        """
        Generate leave-one-out jackknife samples.

        Returns
        -------
        np.ndarray
            Jackknife samples with the same shape as the input data.
        """
        if self._jack_samples is None:
            n = self.ncfg

            data = np.moveaxis(self.data, self.axis, 0)

            total_sum = np.sum(data, axis=0)

            jack_samples = (total_sum[None, ...] - data) / (n - 1)

            self._jack_samples = np.moveaxis(
                jack_samples,
                0,
                self.axis,
            )

        return self._jack_samples

    def mean(self) -> np.ndarray:
        """
        Mean of the resampled data.

        Returns
        -------
        np.ndarray
        """
        return np.mean(self.data, axis=self.axis)

    def error(self) -> np.ndarray:
        """
        Compute jackknife standard error from resampled data.
        Note: Input resampled data here!!!

        Parameters
        ----------
        data : np.ndarray, optional

        Returns
        -------
        np.ndarray
            Jackknife standard error.
        """

        n = self.data.shape[self.axis]

        if n < 2:
            raise ValueError(
                f"Jackknife axis must contain at least 2 samples, got {n}."
            )
        return np.sqrt(
            (n - 1)
            * np.var(
                self.data,
                axis=self.axis,
                ddof=0,
            )
        )

    def gvar(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return mean and jackknife error of the original data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
        """
        return self.mean(), self.error()

    def resample_manual(self, idx: int = 0) -> np.ndarray:
        """
        Generate a leave-one-out dataset by removing one sample along the specified axis.

        Parameters
        ----------
        data : np.ndarray
            Input array containing all samples.
        idx : int
            Index of the sample to remove.
        axis : int
            Axis corresponding to the sample dimension.

        Returns
        -------
        np.ndarray
            Array with the specified sample removed along the sample axis.
        """
        # Validate inputs
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if not (0 <= idx < self.data.shape[self.axis]):
            raise IndexError(
                f"idx must be between 0 and {self.data.shape[self.axis]-1}"
            )

        # Remove the idx-th sample along the sample axis
        return np.delete(self.data, idx, axis=self.axis)
