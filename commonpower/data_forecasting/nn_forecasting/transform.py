"""
Transformation classes for data preprocessing.
"""
from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import torch


class Transformation:
    """
    Transformations are used to preprocess the data before feeding it into the neural network.
    They commonly scale data to certain ranges.
    """

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, type: Literal["feature", "target"], **constructor_kwargs
    ) -> Transformation:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint_path)
        transformer = cls(**constructor_kwargs)
        transformer.load_state_dict(checkpoint[f'{type}_transformer_state_dict'])
        return transformer

    def fit(self, data: np.ndarray):
        """
        Fit the transformation to the data.
        """
        raise NotImplementedError

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data.
        """
        raise NotImplementedError

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the data.
        """
        raise NotImplementedError

    def state_dict(self) -> dict:
        """
        Return the state of the transformation.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the transformation.
        """
        raise NotImplementedError


class IdentityTransform(Transformation):
    def fit(self, data: np.ndarray):
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


class SklearnScalerTransform(Transformation):
    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str, type: Literal["feature", "target"], **constructor_kwargs
    ) -> Transformation:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint_path)
        transformer = cls(**constructor_kwargs)
        transformer.load_state_dict(checkpoint[f'{type}_transformer_state_dict'])
        return transformer

    def __init__(self, scaler: Any):
        """
        Transformation based on a sklearn scaler.
        We need the given scaler to implement fit(), transform() and inverse_transform() methods.
        Caution regarding the state_dict() and load_state_dict() methods:
        Some exotic scalers might need a custom implementation.
        Args:
            scaler (Any, optional): Instance of sklearn scaler.
        """

        self.scaler = scaler

    def fit(self, data: np.ndarray):

        self.scaler = self.scaler.fit(data)

    def __call__(self, data: np.ndarray) -> np.ndarray:

        return self.scaler.transform(data)

    def inverse(self, data: np.ndarray) -> np.ndarray:

        return self.scaler.inverse_transform(data)

    def state_dict(self) -> dict:

        return self.scaler.__dict__

    def load_state_dict(self, state_dict: dict) -> None:

        for key, value in state_dict.items():
            if key not in self.scaler.__dict__:  # we don't overwrite attributes that were set in the constructor
                self.scaler.__dict__[key] = value
