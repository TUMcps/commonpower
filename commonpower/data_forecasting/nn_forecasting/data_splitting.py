"""
Dataset splitting for neural network forecasting.
"""
from __future__ import annotations

from datetime import timedelta
from enum import Enum, auto

from commonpower.data_forecasting.base import DataSource
from commonpower.data_forecasting.nn_forecasting.models import NNModule


class DataSplitType(int, Enum):
    ALL = auto()
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class DatasetSplit:
    def __init__(self, split_type: DataSplitType, data_source: DataSource, model: NNModule) -> DatasetSplit:
        """
        DataSplits are used to split the dataset into training, validation, and test sets.
        They do that by adjusting the index when accessing the dataset.

        Args:
            split_type (DataSplitType): The type of data splitting.
            data_source (DataSource): The data source.
            model (NNModule): The neural network model.
        Returns:
            DatasetSplit: The initialized DatasetSplit object.
        """

        self.type = split_type
        self.data_source = data_source
        self.model = model

    def adjust_index(self, idx: int) -> int:
        """
        Adjusts the index to the correct position in the dataset.

        Args:
            idx (int): Index.

        Returns:
            int: Adjusted index.
        """
        return idx

    def __len__(self) -> int:
        return len(self.data_source)


class SimpleFractionalSplit(DatasetSplit):
    def __init__(
        self,
        split_type: DataSplitType,
        data_source: DataSource,
        model: NNModule,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
    ):
        """
        The SimpleFractionalSplit splits the dataset into training, validation, and test sets based on fractions.
        The split is done in order of the data source (generally date).
        If `train_fraction` + `val_fraction` < 1, the remaining data is used for testing.

        Args:
            split_type (DataSplitType): The type of data splitting.
            data_source (DataSource): The data source.
            model (NNModule): The neural network model.
            train_fraction (float, optional): The fraction of the data to use for training. Defaults to 0.7.
            val_fraction (float, optional): The fraction of the data to use for validation. Defaults to 0.15.
        Returns:
            DatasetSplit: The initialized DatasetSplit object.
        """

        super().__init__(split_type, data_source, model)

        n_look_back = model.input_shape[0] - 1
        n_steps_ahead = model.output_shape[0]

        train_len_native = int(len(data_source) * train_fraction)
        val_len_native = int(len(data_source) * val_fraction)

        self.train_len = train_len_native - n_look_back - n_steps_ahead
        self.val_len = val_len_native - n_look_back - n_steps_ahead

        assert self.train_len > 0, "There is insufficient data for training."
        assert self.val_len > 0, "There is insufficient data for validation."

        test_len = len(data_source) - train_len_native - val_len_native
        self.test_len = test_len - n_look_back - n_steps_ahead if test_len > 0 else 0

        self.train_offset = n_look_back
        self.val_offset = train_len_native + n_look_back
        self.test_offset = train_len_native + val_len_native + n_look_back

    def adjust_index(self, idx: int) -> int:
        if self.type == DataSplitType.TRAIN:
            return idx + self.train_offset
        elif self.type == DataSplitType.VAL:
            return idx + self.val_offset
        elif self.type == DataSplitType.TEST:
            return idx + self.test_offset
        else:
            raise ValueError("Invalid split type.")

    def __len__(self) -> int:
        if self.type == DataSplitType.TRAIN:
            return self.train_len
        elif self.type == DataSplitType.VAL:
            return self.val_len
        elif self.type == DataSplitType.TEST:
            return self.test_len
        else:
            raise ValueError("Invalid split type.")


class DatePeriodFractionalSplit(DatasetSplit):
    def __init__(
        self,
        split_type: DataSplitType,
        data_source: DataSource,
        model: NNModule,
        period_length: timedelta = timedelta(weeks=4),
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
    ):
        """
        This Splitter first divides the dataset into periods of length `period_length` in order of the data source.
        Each period is then split into training, validation, and test sets according to the fractions provided.
        The advantage over the SimpleFractionalSplit is that we avoid bias based on seasonality
        or distribution shift over time.

        Args:
            split_type (DataSplitType): The type of data splitting.
            data_source (DataSource): The data source.
            model (NNModule): The neural network model.
            period_length (timedelta, optional): The length of each period. Defaults to timedelta(weeks=4).
            train_fraction (float, optional): The fraction of the data to use for training. Defaults to 0.7.
            val_fraction (float, optional): The fraction of the data to use for validation. Defaults to 0.15.
        Returns:
            DatasetSplit: The initialized DatasetSplit object.
        """

        super().__init__(split_type, data_source, model)

        self.period_length = period_length

        n_look_back = model.input_shape[0] - 1
        n_steps_ahead = model.output_shape[0]

        n_idxs_per_period = period_length // data_source.frequency
        n_periods = len(data_source) // n_idxs_per_period

        train_len_per_period_native = int(n_idxs_per_period * train_fraction)
        val_len_per_period_native = int(n_idxs_per_period * val_fraction)

        self.train_len_per_period = train_len_per_period_native - n_look_back - n_steps_ahead
        self.val_len_per_period = val_len_per_period_native - n_look_back - n_steps_ahead

        assert self.train_len_per_period > 0, "There is insufficient data for training."
        assert self.val_len_per_period > 0, "There is insufficient data for validation."

        self.test_len_per_period = n_idxs_per_period - train_len_per_period_native - val_len_per_period_native
        self.test_len_per_period = (
            self.test_len_per_period - n_look_back - n_steps_ahead if self.test_len_per_period > 0 else 0
        )

        self.period_offsets = [(i * n_idxs_per_period) for i in range(n_periods)]

        self.train_offset = n_look_back
        self.val_offset = train_len_per_period_native + n_look_back
        self.test_offset = train_len_per_period_native + val_len_per_period_native + n_look_back

        self.n_train = n_periods * self.train_len_per_period
        self.n_val = n_periods * self.val_len_per_period
        self.n_test = n_periods * self.test_len_per_period

    def adjust_index(self, idx: int) -> int:

        if self.type == DataSplitType.TRAIN:
            len_per_period = self.train_len_per_period
            offset_in_period = self.train_offset
        elif self.type == DataSplitType.VAL:
            len_per_period = self.val_len_per_period
            offset_in_period = self.val_offset
        elif self.type == DataSplitType.TEST:
            len_per_period = self.test_len_per_period
            offset_in_period = self.test_offset
        else:
            raise ValueError("Invalid split type.")

        period_idx = idx // len_per_period
        idx_in_period = idx % len_per_period

        return self.period_offsets[period_idx] + idx_in_period + offset_in_period

    def __len__(self) -> int:

        if self.type == DataSplitType.TRAIN:
            return self.n_train
        elif self.type == DataSplitType.VAL:
            return self.n_val
        elif self.type == DataSplitType.TEST:
            return self.n_test
        else:
            raise ValueError("Invalid split type.")
