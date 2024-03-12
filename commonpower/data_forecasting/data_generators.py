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

        complete_schedule = np.tile(is_plugged_in, int((end_date - start_date).days) + 1)

        date_index = pd.date_range(start_date, end_date, freq=frequency)

        data = pd.DataFrame(complete_schedule, index=date_index, columns=["is_plugged_in"])

        return PandasDataSource(data, frequency)
