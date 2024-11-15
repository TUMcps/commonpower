"""
Collection of data generators.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from commonpower.data_forecasting.data_sources import PandasDataSource


class EVDataGenerator:
    def generate_constant_schedule(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: timedelta,
        departure: int,
        arrival: int,
    ) -> PandasDataSource:
        """
        Generate a dataframe of EV charging data for the specified time period.
        Assumes a daily (24h) schedule that repeats every day.
        """

        time_steps_per_day = int(timedelta(days=1) / frequency)

        is_plugged_in = np.zeros((time_steps_per_day,))
        is_plugged_in[:departure] = 1
        is_plugged_in[arrival:] = 1

        departure_indicator = np.zeros((time_steps_per_day,))
        departure_indicator[departure] = 1

        return_indicator = np.zeros((time_steps_per_day,))
        return_indicator[arrival] = 1

        complete_schedule = np.tile(is_plugged_in, int((end_date - start_date).days) + 1)
        complete_departure_indicator = np.tile(departure_indicator, int((end_date - start_date).days) + 1)
        complete_return_indicator = np.tile(return_indicator, int((end_date - start_date).days) + 1)

        date_index = pd.date_range(start_date, end_date, freq=frequency)

        data = pd.DataFrame(
            {
                "is_plugged_in": complete_schedule,
                "departure_indicator": complete_departure_indicator,
                "return_indicator": complete_return_indicator,
            },
            index=date_index,
        )

        return PandasDataSource(data, frequency)
