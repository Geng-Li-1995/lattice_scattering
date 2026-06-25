import numpy as np


class Jackknife:
    """Leave-one-out jackknife resampling."""

    def __init__(self, data: np.ndarray, axis: int = -1):
        self.data = np.asarray(data)
        self.axis = axis
        self._jack_samples = None

    @property
    def ncfg(self) -> int:
        return self.data.shape[self.axis]

    def resample(self) -> np.ndarray:
        if self._jack_samples is None:
            n = self.ncfg
            data = np.moveaxis(self.data, self.axis, 0)
            total_sum = np.sum(data, axis=0)
            jack_samples = (total_sum[None, ...] - data) / (n - 1)
            self._jack_samples = np.moveaxis(jack_samples, 0, self.axis)
        return self._jack_samples

    def mean(self) -> np.ndarray:
        return np.mean(self.data, axis=self.axis)

    def error(self) -> np.ndarray:
        n = self.data.shape[self.axis]
        if n < 2:
            raise ValueError(
                f"Jackknife axis must contain at least 2 samples, got {n}."
            )
        return np.sqrt((n - 1) * np.var(self.data, axis=self.axis, ddof=0))

    def gvar(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean(), self.error()

    def resample_manual(self, idx: int = 0) -> np.ndarray:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if not (0 <= idx < self.data.shape[self.axis]):
            raise IndexError(
                f"idx must be between 0 and {self.data.shape[self.axis]-1}"
            )
        return np.delete(self.data, idx, axis=self.axis)
