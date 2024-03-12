from __future__ import annotations

from datetime import datetime
from typing import Union

import numpy as np


class ParamInitializer:
    """
    Base class. \\
    ParamInitializers specify how parameters are initialized on a system reset.
    This allows for the creation of more diverse scenarios.
    """

    def get_init_val(self, at_time: datetime) -> Union[int, float]:
        """
        Returns initial parameter value.

        Args:
            at_time (datetime): Current timestamp.

        Raises:
            NotImplementedError: Needs to be implemented by subclass.

        Returns:
            Union[int, float]: Initial value.
        """
        raise NotImplementedError


class RangeInitializer(ParamInitializer):
    def __init__(self, lb: Union[int, float], ub: Union[int, float], sampling_mode: str = "uniform"):
        """
        This initializer samples values from a given range.

        Args:
            lb (Union[int, float]): Lower bound.
            ub (Union[int, float]): Upper bound.
            sampling_mode (str, optional): Sampling distribution. If lower and upper bound are given as integers,
                only integers will be sampled. Options: "uniform". Defaults to "uniform".
        """
        self.lb = lb
        self.ub = ub
        self.sampling_mode = sampling_mode

    def get_init_val(self, at_time: datetime) -> Union[int, float]:
        """
        Returns value sampled from specified range.

        Args:
            at_time (datetime): Current timestamp.

        Raises:
            NotImplementedError: If anything but "uniform" was specified as sampling_mode.

        Returns:
            Union[int, float]: Initial value.
        """
        if self.sampling_mode == "uniform":
            if isinstance(self.lb, int) and isinstance(self.ub, int):
                return np.random.randint(self.lb, self.ub)
            else:
                return np.random.uniform(self.lb, self.ub)
        else:
            raise NotImplementedError


class ConstantInitializer(ParamInitializer):
    def __init__(self, val: Union[int, float]):
        """
        This initializer always samples a constant given value.

        Args:
            val (Union[int, float]): Value to be sampled.
        """
        self.val = val

    def get_init_val(self, at_time: datetime) -> Union[int, float]:
        """
        Returns specified constant value.

        Args:
            at_time (datetime): Current timestamp.

        Returns:
            Union[int, float]: Initial value.
        """
        return self.val


class IterableInitializer(ParamInitializer):
    def __init__(self, vals: Union[tuple[float], list[float], tuple[int], list[int]]):
        """
        This initializer iterates over a given iterable.
        Once a full iteration is complete, it starts over.

        Args:
            vals (Union[tuple[float], list[float], tuple[int], list[int]]): Values to iterate over.
        """
        self.vals = vals
        self.n = len(vals)
        self.idx = -1
        self.time = datetime()

    def get_init_val(self, at_time: datetime) -> Union[int, float]:
        """
        Returns the value from the current position in the iterable.

        Args:
            at_time (datetime): Current timestamp.
                If the given timestamp is smaller than the timestamp that was given at the last call,
                the iteration resets to the beginning.

        Returns:
            float: Initial value.
        """
        if at_time < self.time or self.n <= self.idx + 1:
            self.idx = -1

        self.idx += 1
        self.time = at_time

        return self.vals[self.idx]
