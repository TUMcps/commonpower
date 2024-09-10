"""
Dataset wrappers for neural network forecasting.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from commonpower.data_forecasting.base import DataSource
from commonpower.data_forecasting.nn_forecasting.data_splitting import DatasetSplit
from commonpower.data_forecasting.nn_forecasting.models import NNModule
from commonpower.data_forecasting.nn_forecasting.transform import IdentityTransform, Transformation


class DatasetWrapper(Dataset):
    def __init__(
        self,
        data_source: DataSource,
        data_split: DatasetSplit,
        model: NNModule,
        targets: list[str],
        features: list[str],
        feature_transform: Transformation = IdentityTransform(),
        target_transform: Transformation = IdentityTransform(),
    ) -> DatasetWrapper:
        """
        DatasetWrappers determine how the train/val datasets are constructed.
        It acts as a wrapper around the data source and can be passed to a data loader as dataset.

        Args:
            data_source (DataSource): The data source for the dataset.
            data_split (DatasetSplit): The split of the dataset.
            model (NNModule): The neural network model.
            targets (list[str]): The list of target variables.
            features (list[str]): The list of feature variables.
            feature_transform (Transformation, optional): The transformation to apply to the features.
                Defaults to IdentityTransform().
            target_transform (Transformation, optional): The transformation to apply to the targets.
                Defaults to IdentityTransform().
        Returns:
            DatasetWrapper: The initialized DatasetWrapper object.
        """

        self.data_source = data_source
        self.model = model
        self.data_split = data_split

        self.feature_transform = feature_transform
        self.target_transform = target_transform

        self.feature_idxs = [i for i, var in enumerate(self.data_source.get_variables()) if var in features]
        self.target_idxs = [i for i, var in enumerate(self.data_source.get_variables()) if var in targets]

        return self

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset.
        """
        return len(self.data_split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data point at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (input, target)
        """
        return self._get_item(self.data_split.adjust_index(idx))

    def _get_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class NStepAhead(DatasetWrapper):
    def __init__(
        self,
        data_source: DataSource,
        data_split: DatasetSplit,
        model: NNModule,
        targets: list[str],
        features: list[str],
        feature_transform: Transformation = IdentityTransform(),
        target_transform: Transformation = IdentityTransform(),
    ) -> NStepAhead:
        """
        The NStepAhead wrapper inspects the passed model instance and
        determines the look back and steps ahead values from the model input and output shapes.
        The dataset is then constructed such that each data point has input dimension (n_look_back, n_features)
        and target dimension (n_steps_ahead, n_targets).

        Args:
            data_source (DataSource): The data source for the dataset.
            data_split (DatasetSplit): The split of the dataset.
            model (NNModule): The neural network model.
            targets (list[str]): The list of target variables.
            features (list[str]): The list of feature variables.
            feature_transform (Transformation, optional): The transformation to apply to the features.
                Defaults to IdentityTransform().
            target_transform (Transformation, optional): The transformation to apply to the targets.
                Defaults to IdentityTransform().
        Returns:
            NStepAhead: The initialized NStepAhead object.
        """

        super().__init__(data_source, data_split, model, targets, features, feature_transform, target_transform)

        self.steps_ahead = model.output_shape[0]
        self.look_back = model.input_shape[0] - 1  # we are also considering the current time step in the model input

    def _get_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data point at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (input, target)
        """
        idx_time = self.data_source.get_date_range()[0] + idx * self.data_source.frequency
        feature_start_time = idx_time - self.look_back * self.data_source.frequency

        target_start_time = idx_time + self.data_source.frequency
        target_end_time = idx_time + self.steps_ahead * self.data_source.frequency

        input_data = self.data_source(feature_start_time, idx_time)[:, self.feature_idxs]
        target_data = self.data_source(target_start_time, target_end_time)[:, self.target_idxs]

        input_data_transformed = self.feature_transform(input_data)
        target_data_transformed = self.target_transform(target_data)

        return torch.tensor(input_data_transformed).float(), torch.tensor(target_data_transformed).float()
