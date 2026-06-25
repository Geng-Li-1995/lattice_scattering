import numpy as np


class Bootstrap:
    """Bootstrap resampling with replacement."""

    def __init__(
        self,
        data: np.ndarray,
        axis: int = -1,
        nboot: int = 1000,
        random_seed: int | None = None,
    ):
        self.data = np.asarray(data)
        self.axis = axis
        self.nboot = nboot
        self.random_seed = random_seed
        self._boot_samples = None
        self._rng = np.random.default_rng(random_seed)

    @property
    def ncfg(self) -> int:
        return self.data.shape[self.axis]

    def resample(self) -> np.ndarray:
        if self._boot_samples is None:
            n = self.ncfg
            data = np.moveaxis(self.data, self.axis, 0)
            indices = self._rng.integers(0, n, size=(self.nboot, n))
            self._boot_samples = np.moveaxis(data[indices], 1, self.axis + 1)
        return self._boot_samples

    def mean(self) -> np.ndarray:
        return np.mean(self.data, axis=self.axis)

    def error(self) -> np.ndarray:
        return np.std(self.resample(), axis=self.axis, ddof=0)

    def gvar(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean(), self.error()

    def resample_manual(self, idx: np.ndarray | None = None) -> np.ndarray:
        n = self.ncfg
        if idx is None:
            idx = self._rng.integers(0, n, size=n)
        elif not isinstance(idx, np.ndarray):
            idx = np.asarray(idx)
        return np.take(self.data, idx, axis=self.axis)
