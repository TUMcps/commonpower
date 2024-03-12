"""
Collection of forecasters.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Union

import numpy as np

from commonpower.data_forecasting.base import Forecaster


class ConstantForecaster(Forecaster):
    def __init__(self, frequency: timedelta = timedelta(hours=1), horizon: timedelta = timedelta(hours=24)):
        """
        This forecaster predicts all future timesteps with the "current" value.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
        """
        super().__init__(frequency, horizon, timedelta())

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.repeat(data, int(self.horizon / self.frequency), axis=0)


class PersistenceForecaster(Forecaster):
    def __init__(
        self,
        frequency: timedelta = timedelta(hours=1),
        horizon: timedelta = timedelta(hours=24),
        look_back: timedelta = timedelta(hours=24),
    ):
        """
        This forecaster predicts all future timesteps with the value look_back before, i.e.,
        every value is predicted as the value at time t-look_back.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
            look_back (timedelta, optional): Look back time to use for predictions. Defaults to timedelta(hours=24).
        """
        super().__init__(frequency, horizon, look_back)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data[1 : int(self.horizon / self.frequency) + 1, :]


class LookBackForecaster(Forecaster):
    def __init__(self, frequency: timedelta = timedelta(hours=1), horizon: timedelta = timedelta(hours=24)):
        """
        This forecaster predicts the last timestep of the horizon as the "current" value.
        Every timestep t until then is predicted as the value at time t-horizon.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
        """
        super().__init__(frequency, horizon, horizon - frequency)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data


class PerfectKnowledgeForecaster(Forecaster):
    def __init__(self, frequency: timedelta = timedelta(hours=1), horizon: timedelta = timedelta(hours=24)):
        """
        This forecaster perfectly predicts future values.
        This means all time steps in the prediction horizon must be present in the data source.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
        """
        super().__init__(frequency, horizon, timedelta())

    def input_range(self) -> tuple[timedelta]:
        return (self.frequency, self.horizon)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data


class NoisyForecaster(Forecaster):
    def __init__(
        self,
        frequency: timedelta = timedelta(hours=1),
        horizon: timedelta = timedelta(hours=24),
        noise_bounds: Union[float, list[float]] = [-0.1, 0.1],
    ):
        """
        This forecaster knows the true future values but applies a uniformly random noise to it.
        This means all time steps in the prediction horizon must be present in the data source.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
            noise_bounds(Union[float, list[float]], optional): Lower and upper relative noise bounds.
                Defaults to [-0.1, 0.1].
        """
        super().__init__(frequency, horizon, timedelta())

        assert noise_bounds[0] <= noise_bounds[1], "Lower noise bound must be lower than upper bound."
        self.b = noise_bounds

    def input_range(self) -> tuple[timedelta]:
        return (self.frequency, self.horizon)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.array([d + np.random.uniform(low=(abs(d) * self.b[0]), high=(abs(d) * self.b[1])) for d in data])
