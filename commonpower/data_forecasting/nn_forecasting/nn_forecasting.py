"""
NNForecaster class.
"""
from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import torch
from numpy import concatenate, ndarray
from torch.utils.data import DataLoader

from commonpower.data_forecasting.base import DataSource, Forecaster
from commonpower.data_forecasting.nn_forecasting.data_splitting import (
    DatasetSplit,
    DataSplitType,
    SimpleFractionalSplit,
)
from commonpower.data_forecasting.nn_forecasting.dataset_wrappers import DatasetWrapper, NStepAhead
from commonpower.data_forecasting.nn_forecasting.models import NNModule
from commonpower.data_forecasting.nn_forecasting.transform import IdentityTransform, Transformation

if TYPE_CHECKING:
    from commonpower.data_forecasting.nn_forecasting.config import ParameterSpace


class NNForecaster(Forecaster):
    def __init__(
        self,
        model_class: NNModule.__class__,
        targets: list[str],
        frequency: timedelta = timedelta(hours=1),
        horizon: timedelta = timedelta(hours=12),
        feature_transform: Transformation = IdentityTransform(),
        target_transform: Transformation = IdentityTransform(),
    ):
        """
        Neural-Network-based Forecaster.

        All featues of the data source (including targets) will be used as model inputs.
        We make the assumption that all features besides the targets are static in the sense that they
        are available across the entire forecast horizon (e.g. time features).
        This is is necessary to apply the model iteratively.
        If this assumption cannot reasonably made in practice,
        the model output must cover the entire horizon in one step.

        When the forecaster is deployed, we assume that the targets are the first "columns"
        of the data source.

        Args:
            model_class (NNModule.__class__): Model class.
            targets (list[str]): Target variables.
            frequency (timedelta, optional): Frequency of the data. Defaults to timedelta(hours=1).
            horizon (timedelta, optional): Forecast horizon. Defaults to timedelta(hours=12).
            feature_transform (Transformation, optional): Feature transformation. Defaults to IdentityTransform().
            target_transform (Transformation, optional): Target transformation. Defaults to IdentityTransform().
        """

        self.frequency = frequency
        self.horizon = horizon

        self.model_class = model_class
        self.targets = targets
        self.feature_transform = feature_transform
        self.target_transform = target_transform

        self.model: NNModule = None
        self.model_output_steps: int = None
        self.iteration_steps: int = None

    @property
    def look_back(self) -> timedelta:
        # Model input includes current time step
        return (self.model.input_shape[0] - 1) * self.frequency

    @property
    def input_range(self) -> tuple[timedelta]:
        """
        Returns the min and max timedelta of observations which are required for the prediction.
        To indicate a timestamp before the current time, the timedelta must be negative.

        Returns:
            tuple[timedelta]: (td before, td after)
        """
        return (-self.look_back, self.horizon)

    def with_model(self, model: NNModule) -> NNForecaster:
        """
        This can be called to pass an already trained model to the forecaster.
        We expect the transformations passed in the contructor to be already fitted.

        Args:
            model (NNModule): Forecast model.
        """
        self.model = model
        self.model_output_steps = self.model.output_shape[0]
        self.iteration_steps = (self.horizon // self.frequency) // self.model_output_steps
        return self

    def setup(
        self,
        data_source: DataSource,
        param_space: ParameterSpace,
    ) -> NNForecaster:
        """
        Setup the forecaster for training.
        This is usually called from the NNTrainer.
        This means anything passed to the setup method can be tuned.
        Here, we check some model dimensions and fit the transformations.

        Args:
            data_source (DataSource): Data source.
            param_space (ParameterSpace): Parameter space for the forecaster.
        Returns:
            NNForecaster: The setup forecaster.
        """
        self.param_space = param_space

        self.model: NNModule = self.model_class(**param_space.model)

        # default features are all variables (including targets)
        self.features = data_source.get_variables()

        assert (self.horizon // self.frequency) % self.model.output_shape[
            0
        ] == 0, "Model output shape does not match the horizon."
        assert self.model.output_shape[1] == len(
            self.targets
        ), "Model output shape does not match the number of target dimensions."
        assert self.model.input_shape[1] == len(
            self.features
        ), "Model input shape does not match the number of features."
        assert (
            self.model.input_shape[0] == (-self.input_range[0]) // self.frequency + 1
        ), "Model input shape does not match the forecaster input range."

        # If the model output is lower than the number of prediction steps, we apply the model iteratively
        self.model_output_steps = self.model.output_shape[0]
        self.iteration_steps = (self.horizon // self.frequency) // self.model_output_steps

        # Fit the transformations

        complete_data = data_source(*data_source.get_date_range())
        feature_data = complete_data[
            :, [i for i, var in enumerate(data_source.get_variables()) if var in self.features]
        ]

        self.target_idxs = [i for i, var in enumerate(data_source.get_variables()) if var in self.targets]
        target_data = complete_data[:, self.target_idxs]

        self.feature_transform.fit(feature_data)
        self.target_transform.fit(target_data)

        return self

    def __call__(self, data: ndarray) -> ndarray:
        """
        Make a prediction.
        If the model prediction horizon (steps ahead) are less than the forecast horizon,
        we iteratively apply the model to make predictions covering the entire horizon.
        For this to work, the model feature and target variables must be identical.

        Args:
            data (ndarray): Input data. Expected shape: (N, n_features).
        Returns:
            ndarray: Forecasted values. Shape: (N, n_targets).
        """
        assert self.model is not None, "Model is not set. Call setup() or with_model() first."

        # Apply transformations
        data = self.feature_transform(data)

        # Convert to torch tensor
        data: torch.Tensor = torch.tensor(data).float()

        # Inital data is [-look_back, 0]
        tmp_data = data[: self.model.input_shape[0], :]

        # Make prediction
        # The reshape is necessary because the model expects a batch dimension
        prediction: torch.Tensor = self.model(tmp_data.reshape(1, *tmp_data.shape)).reshape(self.model_output_steps, 1)

        # Apply inverse transformation
        out_prediction: ndarray = self.target_transform.inverse(prediction.detach().cpu().numpy())

        # For each iteration step, we
        # apply input transformation to prediction
        # step data forward by one step
        # replace the targets with the prediction
        # make a new prediction
        # apply the inverse transformation
        tmp_prediction = out_prediction
        for t in range(1, self.iteration_steps):
            tmp_prediction: torch.Tensor = torch.tensor(self.target_transform(tmp_prediction)).float()

            tmp_data = data[t * self.model_output_steps : self.model.input_shape[0] + t * self.model_output_steps, :]
            # we assume target variables are the first columns
            tmp_data[-self.model_output_steps :, : tmp_prediction.shape[1]] = tmp_prediction

            tmp_prediction: torch.Tensor = self.model(tmp_data.reshape(1, *tmp_data.shape)).reshape(
                self.model_output_steps, 1
            )
            tmp_prediction: ndarray = self.target_transform.inverse(tmp_prediction.detach().cpu().numpy())

            out_prediction = concatenate((out_prediction, tmp_prediction), axis=0)

        return out_prediction

    def get_train_val_loaders(
        self,
        data_source: DataSource,
        param_space: ParameterSpace,
        dataset_wrapper_class: DatasetWrapper.__class__ = NStepAhead,
        dataset_split_class: DatasetSplit.__class__ = SimpleFractionalSplit,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Return training and validation data loaders.

        Args:
            data_source (DataSource): Data source.
            param_space (ParameterSpace): Parameter space for the forecaster.
            dataset_wrapper_class (DatasetWrapper.__class__, optional): Dataset wrapper class. Defaults to NStepAhead.
            dataset_split_class (DatasetSplit.__class__, optional): Dataset split class.
                Defaults to SimpleFractionalSplit.

        Returns:
            tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """
        train_dataset = dataset_wrapper_class(
            data_source,
            dataset_split_class(
                DataSplitType.TRAIN,
                data_source,
                self.model,
                **param_space.dataset_split,
            ),
            self.model,
            self.targets,
            self.features,
            self.feature_transform,
            self.target_transform,
            **param_space.dataset_wrapper,
        )

        val_dataset = dataset_wrapper_class(
            data_source,
            dataset_split_class(
                DataSplitType.VAL,
                data_source,
                self.model,
                **param_space.dataset_split,
            ),
            self.model,
            self.targets,
            self.features,
            self.feature_transform,
            self.target_transform,
            **param_space.dataset_wrapper,
        )

        train_loader = DataLoader(train_dataset, **param_space.data_loader)
        val_loader = DataLoader(val_dataset, **param_space.data_loader)

        return train_loader, val_loader
