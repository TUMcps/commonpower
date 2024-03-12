"""
Collection of data sources.
"""
from __future__ import annotations

from copy import copy
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from commonpower.data_forecasting.base import DataSource


class PandasDataSource(DataSource):
    def __init__(self, data: pd.DataFrame, frequency: timedelta = timedelta(hours=1)):
        """
        Data source based on a pandas dataframe.

        Args:
            data (pd.DataFrame): Dataframe containing the data.
                The index needs to be a datetime index.
            frequency (timedelta, optional): Frequency of the data. Defaults to timedelta(hours=1).
        """
        self.frequency = frequency
        self.data = data

    def get_date_range(self) -> List[datetime]:
        return copy([self.data.index[0].to_pydatetime(), self.data.index[-1].to_pydatetime()])

    def get_variables(self) -> List[str]:
        return self.data.columns.to_numpy()

    def apply_to_column(self, column: str, fcn: callable) -> DataSource:
        """
        Allows applying a transformation to a column of the data (using pandas df.apply()).

        Args:
            column (str): Column to manipulate.
            fcn (callable): Transformation to apply.
                The fcn needs to take one argument which refers to the value of a cell: fcn(x).

        Returns:
            DataSource: self
        """
        self.data[column] = self.data[column].apply(fcn)
        return self

    def shift_time_series(self, shift_by: timedelta) -> DataSource:
        """
        Shifts time series by a given timedelta.
        The shift is done in a rolling fashing such that the start and end timestamps do not change.
        Can be used to simulate more diverse data.

        Args:
            shift_by (timedelta): Time delta to shift by.
                Posititve values shift into the "future", negative into the "past".

        Returns:
            DataSource: self
        """
        shift_steps = int(shift_by / self.frequency)

        self.data = pd.DataFrame(
            np.roll(self.data.values, shift=shift_steps, axis=0), index=self.data.index, columns=self.data.columns
        )
        return self

    def __call__(self, from_time: datetime, to_time: datetime) -> np.ndarray:
        return self.data.loc[from_time:to_time].to_numpy()


class CSVDataSource(PandasDataSource):
    def __init__(
        self,
        file_path: str,
        datetime_format: str = "%d.%m.%Y %H:%M",
        rename_dict: dict = {},
        auto_drop: bool = False,
        resample: timedelta = timedelta(hours=1),
        aggregation_method: str = "mean",
        aggregation_alignment: str = "future",
        interpolation_method: str = "time",
        **csv_read_kwargs,
    ) -> CSVDataSource:
        """
        DataSource based on .csv data.
        It imports from a .csv file, does some preprocessing and stores it in an internal pandas data frame.

        Args:
            file_path (str): Path to the source .csv file.
            datetime_format (_type_, optional): Datetime format the source .csv file.
                Specifically, this refers to the (required) column "t". Defaults to "%d.%m.%Y %H:%M".
            rename_dict (dict, optional): Dict to specify column renaming. Format: {"original name": "new name", ...}.
                Defaults to {}.
            auto_drop (bool, optional): If set to true, all columns of the source data except those mentioned in
                rename_dict will be dropped. Defaults to False.
            resample (timedelta, optional): Resamples the source data to this value. If the time interval of
                the source data is larger than the resample value, the data is interpolated linearly.
                Defaults to timedelta(hours=1).
            aggregation_method (str, optional): Method by which to aggregate multiple source datapoints for one sample.
                Parameter to pandas Resampler.aggregate.
                e.g. 'mean', 'last', 'first', 'max'
                Defaults to 'mean'
            aggregation_alignment (str, optional): How to align the resampled windows.
                For easier use, aliases 'future' and 'past' are provided
                Parameter to pandas resample(label=, closed=)
                'left' ('future') or 'right' ('past')
                Defaults to 'future'
            interpolation_method (str, optional): Method by which to interpolate fewer source datapoints to more samples
                Parameter to pandas interpolate()
                Defaults to 'time'
        """
        self.frequency = resample

        self.data = pd.read_csv(file_path, **csv_read_kwargs).rename(columns=rename_dict)
        self.datetime_format = datetime_format

        assert "t" in self.data.columns, (
            "The data needs a time column called 't'. Alternatively, you can specify the name of the time column in"
            " the rename_dict"
        )
        assert not self.data.isnull().any().any(), "There were NaN entries in the data"

        if auto_drop is True:
            # automatically drop all columns that were not mentioned in rename_dict
            self.data.drop(self.data.columns.difference([col for col in rename_dict.values()]), axis=1, inplace=True)

        # make time column the index
        self.data.t = self.data.t.apply(lambda x: datetime.strptime(x, datetime_format))
        self.data.set_index("t", inplace=True, verify_integrity=True)

        # translate aliases for aggregation_alignment (left and right are hard to understand)
        aggregation_alignment = {'future': 'left', 'past': 'right'}.get(aggregation_alignment, aggregation_alignment)

        # resample data
        self.data = (
            self.data.resample(resample, label=aggregation_alignment, closed=aggregation_alignment)
            .agg(aggregation_method)
            .interpolate(interpolation_method)
        )


class ArrayDataSource(DataSource):
    def __init__(
        self,
        values_dict: dict,
        date_range: List[datetime],
        frequency: timedelta = timedelta(hours=1),
        start_date: Optional[datetime] = None,
    ):
        """
        DataSource which repeatedly returns array values, one for each timestep.

        Usage Example:
        ```
        looping_data = ArrayDataSource({
            "day_night": [0.] * 6 + [1.] * 12 + [0.] * 6,
            "weekends": [0.] * 24 * 5 + [1.] * 24 * 2,
            "prime_chaos": [0, 42, 73, 4, 5, 6, 7],
            },
            date_range=[pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31')]  # 2024 starts with a Monday
        )
        ```

        Args:
            values_dict (dict): Dict containing element names and the respective arrays that should be returned.
            date_range (List[datetime]): Date range to simulate.
            frequency (timedelta, optional): Frequency of data to simulate. Defaults to timedelta(hours=1).
            start_date (datetime, optional): The datetime at which the arrays begin looping. Defaults to date_range[0]
        """
        super().__init__(frequency)
        self.values_dict = values_dict
        self.date_range = date_range
        self.start_date = start_date or date_range[0]

    def get_variables(self) -> List[str]:
        return list(self.values_dict.keys())

    def __call__(self, from_time: datetime, to_time: datetime) -> np.ndarray:
        n_start = int((from_time - self.start_date) / self.frequency)
        n_steps = int((to_time - from_time) / self.frequency) + 1

        n_vars = len(self.values_dict)
        out = np.empty([n_steps, n_vars])
        for i, values in enumerate(self.values_dict.values()):
            ln = len(values)
            repeated = np.tile(values, int(n_steps / ln) + 2)
            offset = n_start % ln
            out[:, i] = repeated[offset : n_steps + offset]
        return out

    def get_date_range(self) -> List[datetime]:
        return self.date_range


class ConstantDataSource(DataSource):
    def __init__(self, values_dict: dict, date_range: List[datetime], frequency: timedelta = timedelta(hours=1)):
        """
        Dummy DataSource which returns constant values.

        Args:
            values_dict (dict): Dict containing element names and the respective constant value that should be returned.
            date_range (List[datetime]): Date range to simulate.
            frequency (timedelta, optional): Frequency of data to simulate. Defaults to timedelta(hours=1).
        """
        self.frequency = frequency
        self.values_dict = values_dict
        self.date_range = date_range

    def get_variables(self) -> List[str]:
        return list(self.values_dict.keys())

    def __call__(self, from_time: datetime, to_time: datetime) -> np.ndarray:
        n_steps = int((to_time - from_time) / self.frequency) + 1
        return np.repeat(np.array(list(self.values_dict.values())).reshape((1, -1)), n_steps, axis=0)

    def get_date_range(self) -> List[datetime]:
        return self.date_range
