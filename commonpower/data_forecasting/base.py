"""
Base classes and generic functionality for data sources and forecasters.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Union

import numpy as np

from commonpower.utils import to_datetime


class DataSource:
    def __init__(self, frequency: timedelta = timedelta(hours=1)):
        """
        Data source.
        Data sources manage data for e.g. renewable power generation, demand, or market prices.

        Args:
            frequency (timedelta, optional): Frequency of the data. Defaults to timedelta(hours=1).
        """
        self.frequency = frequency

    def get_variables(self) -> List[str]:
        """
        Returns the list of element names that data is available for.

        Returns:
            List[str]: List of available elements.
        """
        raise NotImplementedError

    def __call__(self, from_time: datetime, to_time: datetime) -> np.ndarray:
        """
        Return the data in this date range.

        Args:
            from_time (datetime): Start time of observation.
            to_time (datetime): End time of observation.

        Returns:
            np.ndarray: Data of shape (n_horizon, n_vars).
        """
        raise NotImplementedError

    def get_date_range(self) -> List[datetime]:
        """
        Returns the date range data is available for.

        Returns:
            List[datetime]: [start_date, end_date]
        """
        raise NotImplementedError


class Forecaster:
    def __init__(
        self,
        frequency: timedelta = timedelta(hours=1),
        horizon: timedelta = timedelta(hours=24),
        look_back: timedelta = timedelta(),
    ):
        """
        Forecaster.

        Args:
            frequency (timedelta, optional): Frequency of generated forecasts. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Horizon to generate forecasts for. Defaults to timedelta(hours=24).
            look_back (timedelta, optional): Amount of time to look into the past for forecast generation.
                Defaults to timedelta().
        """
        assert horizon % frequency == timedelta(), "Forecast horizon must be an integer multiple of the frequency"
        assert look_back % frequency == timedelta(), "Look back time must be an integer multiple of the frequency"

        self.frequency = frequency
        self.horizon = horizon
        self.look_back = look_back

    def input_range(self) -> tuple[timedelta]:
        """
        Returns the min and max timedelta of observations which are required for the prediction.
        To indicate a timestamp before the current time, the timedelta must be negative.
        The default is (-self.look_back, 0).

        Returns:
            tuple[timedelta]: (td before, td after)
        """
        return (-self.look_back, timedelta())

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Returns the forecast based on this data.

        Args:
            data (np.ndarray): Data to use for the forecast.

        Returns:
            np.ndarray: Forecast of shape (n_horizon, n_vars)
        """
        raise NotImplementedError


class DataProvider:
    def __init__(self, data_source: DataSource, forecaster: Forecaster):
        """
        DataProviders combine a DataSource with a Forecaster.

        Args:
            data_source (DataSource): Data source to obtain data from.
            forecaster (Forecaster): Forecaster used for predictions.
        """
        assert forecaster.frequency == data_source.frequency, "Forecaster and data source must have the same frequency"

        self.data = data_source
        self.forecaster = forecaster

        self.horizon = forecaster.horizon
        self.frequency = forecaster.frequency

    def get_variables(self) -> List[str]:
        """
        Returns the list of element names that data is available for.

        Returns:
            List[str]: List of available elements.
        """
        return self.data.get_variables()

    def get_date_range(self) -> List[datetime]:
        """
        Returns the date range data is available for.

        Returns:
            List[datetime]: [start_date, end_date]
        """
        return self.data.get_date_range()

    def _get_current_obs_and_forecast(self, time: datetime) -> List[np.ndarray, np.ndarray]:
        current_obs = self.data(time, time)

        fc_input_range = self.forecaster.input_range()
        fc_input = self.data(time + fc_input_range[0], time + fc_input_range[1])

        fc = self.forecaster(fc_input)

        return current_obs, fc

    def observe(self, time: Union[str, datetime]) -> dict[str, np.ndarray]:
        """
        Returns the observations for all variables of the data provider.
        The observations span the forecast horizon.

        Args:
            time (Union[str, datetime]): Current time.

        Returns:
            dict: {"<element1>": np.ndarray, "<element2>": np.ndarray}.
        """

        time = to_datetime(time)

        current_obs, fc = self._get_current_obs_and_forecast(time)

        out = np.concatenate([current_obs, fc])

        obs_dict = {var: out[:, i] for i, var in enumerate(self.data.get_variables())}

        return obs_dict

    def observation_bounds(self, time: Union[str, datetime]) -> dict[str, tuple(np.ndarray)]:
        """
        Returns the observation bounds for all elements in the data source.
        The default is "guaranteed least-conservative bounds", i.e., the bounds are based on the absolute difference
        between forecast and true value. This only works if the true data is available of course.

        The returned bounds span the forecast horizon.

        Args:
            time (Union[str, datetime]): Current time.

        Returns:
            dict (dict[str, tuple(np.ndarray)]): {"element1": (lower bounds, upper bounds), \
                "element2": (lower bounds, upper bounds)}
        """

        time = to_datetime(time)

        current_obs, fc = self._get_current_obs_and_forecast(time)

        truth = self.data(time + self.frequency, time + self.horizon)

        out = {}

        for i, var in enumerate(self.data.get_variables()):
            lb_var = fc[:, i] - abs(truth[:, i] - fc[:, i])
            ub_var = fc[:, i] + abs(truth[:, i] - fc[:, i])

            out[var] = (np.concatenate([current_obs[:, i], lb_var]), np.concatenate([current_obs[:, i], ub_var]))

        return out
