"""
Base classes and generic functionality for data sources and forecasters.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np


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

    def get_date_range(self) -> List[datetime]:
        """
        Returns the date range data is available for.

        Returns:
            List[datetime]: [start_date, end_date]
        """
        raise NotImplementedError

    def get_limits(self) -> dict[str, tuple[float, float]]:
        """
        Returns the limits for each variable in the data source.

        Returns:
            dict[str, tuple[float, float]]: {"element1": (lower_bound, upper_bound),
                "element2": (lower_bound, upper_bound)}
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

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset.
        """
        raise NotImplementedError


class Forecaster:

    is_uncertain = True  # Generally, forecasts are uncertain

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

    @property
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
    def __init__(self, data_source: DataSource, forecaster: Forecaster, observable_features: list[str] = None):
        """
        DataProviders combine a DataSource with a Forecaster.

        Args:
            data_source (DataSource): Data source to obtain data from.
            forecaster (Forecaster): Forecaster used for predictions.
            observable_features (list[str], optional): List of features from the data source that are observable.
                All other features are only used by the forecaster.
                If not given, all existing features are observed.
                Forecasts can only be generated for observable features. The forecaster must implement that accordingly.
                Defaults to None.
        """
        assert forecaster.frequency == data_source.frequency, "Forecaster and data source must have the same frequency"

        self.data = data_source
        self.forecaster = forecaster

        self.horizon = forecaster.horizon
        self.frequency = forecaster.frequency
        self.observable_features = (
            observable_features if observable_features is not None else data_source.get_variables()
        )

        self.last_provided_data: dict[str, tuple[np.ndarray]] = {}  # Stores the last provided data (obs + forecast)

        self.perfect_knowledge_override: bool = False

    def empty_copy(self) -> DataProvider:
        """
        Returns a copy of the DataProvider.

        Returns:
            DataProvider: Copy of the DataProvider.
        """
        return DataProvider(self.data, self.forecaster, self.observable_features)

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

    def _clip_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Clip the observations to the limits of the data source.

        Args:
            obs (dict[str, np.ndarray]): Observations to clip.

        Returns:
            dict[str, np.ndarray]: Clipped observations.
        """
        limits = self.data.get_limits()

        out = {}

        for var, data in obs.items():
            lb, ub = limits[var]
            out[var] = np.clip(data, lb, ub)

        return out

    def _filter_observed_features(self, data: np.ndarray) -> np.ndarray:
        """
        Filter the data to only include observable features.

        Args:
            data (np.ndarray): Data to filter.

        Returns:
            np.ndarray: Filtered data.
        """
        if self.observable_features is None:
            return data

        indices = [i for i, var in enumerate(self.data.get_variables()) if var in self.observable_features]

        return data[:, indices]

    def _get_current_obs_and_forecast(self, time: datetime) -> List[np.ndarray, np.ndarray]:
        """
        Returns the current observations and the forecast for the current time.

        Args:
            time (datetime): Current time.

        Returns:
            List[np.ndarray, np.ndarray]: [current_obs, forecast]
        """

        # check if we have already provided data for this time
        # mainly used if we need to compute forecast bounds
        if time in self.last_provided_data:
            return self.last_provided_data[time]

        current_obs = self._filter_observed_features(self.data(time, time))

        if not self.perfect_knowledge_override:
            fc_input_range = self.forecaster.input_range
            fc_input = self.data(time + fc_input_range[0], time + fc_input_range[1])

            fc = self.forecaster(fc_input)
        else:
            fc = self.data(time + self.frequency, time + self.horizon)

        self.last_provided_data = {time: (current_obs, fc)}

        return current_obs, fc

    def set_perfect_knowledge(self, perfect_knowledge_active: bool = False):
        """
        Activate or deactivate perfect knowledge override.

        Args:
            perfect_knowledge_active (bool, optional): State of the override.
                Defaults to False.
        """
        self.perfect_knowledge_override = perfect_knowledge_active

    def observe(self, time: datetime) -> dict[str, np.ndarray]:
        """
        Returns the observations for all variables of the data provider.
        The observations span the forecast horizon.
        If the forecaster returns values outside the limits of the data source,
            they are clipped.

        Args:
            time (datetime): Current time.

        Returns:
            dict: {"<element1>": np.ndarray, "<element2>": np.ndarray}.
        """

        current_obs, fc = self._get_current_obs_and_forecast(time)

        out = np.concatenate([current_obs, fc])

        obs_dict = {var: out[:, i] for i, var in enumerate(self.observable_features)}

        return self._clip_obs(obs_dict)

    def observation_bounds(self, time: datetime) -> dict[str, list[tuple[float]]]:
        """
        Returns the observation bounds for all elements in the data source.
        The default is "guaranteed symmetrical bounds", i.e., the bounds are based on the absolute difference
        between forecast and true value.
        Accordingly, the true value is always either the upper or lower limit of the forecast bounds.
        This only works if the true data is available.
        We return bounds for each time step in the forecast horizon.

        Args:
            time (datetime): Current time.

        Returns:
            dict (dict[str, list[tuple[float]]]): {"element1": [(lb_0, ub_0), (lb_1, ub_1)], \
                "element2": [(lb_0, ub_0), (lb_1, ub_1)]}
        """

        current_obs, fc = self._get_current_obs_and_forecast(time)

        truth = self.data(time + self.frequency, time + self.horizon)

        out = {}
        limits = self.data.get_limits()

        for i, var in enumerate(self.observable_features):
            lb_var = fc[:, i] - abs(truth[:, i] - fc[:, i])
            ub_var = fc[:, i] + abs(truth[:, i] - fc[:, i])

            lb_var = np.clip(lb_var, limits[var][0], limits[var][1])
            ub_var = np.clip(ub_var, limits[var][0], limits[var][1])

            out[var] = [
                x for x in zip(np.concatenate([current_obs[:, i], lb_var]), np.concatenate([current_obs[:, i], ub_var]))
            ]

        return out
