"""
Evaluation metrics for neural network forecasting.
"""
import torch
from pydantic import BaseModel
from torch import Tensor


class EvalMetric(BaseModel):
    """
    Base class for evaluation metrics.
    """

    name: str

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the evaluation metric.

        Args:
            y_true (Tensor): The ground truth values. Expected shape: (N, 1).
            y_pred (Tensor): The predicted values. Expected shape: (N, C), with N samples and C classes.

        Returns:
            float: The calculated metric value.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError


class MeanAbsoluteError(EvalMetric):
    """
    Mean Absolute Error (MAE) evaluation metric.
    """

    name: str = 'mae'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Mean Absolute Error (MAE).

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The MAE value.
        """
        return torch.mean(torch.abs(y_true - y_pred)).item()


class MeanSquaredError(EvalMetric):
    """
    Mean Squared Error (MSE) evaluation metric.
    """

    name: str = 'mse'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Mean Squared Error (MSE).

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The MSE value.
        """
        return torch.mean((y_true - y_pred) ** 2).item()


class RootMeanSquaredError(EvalMetric):
    """
    Root Mean Squared Error (RMSE) evaluation metric.
    """

    name: str = 'rmse'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE).

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The RMSE value.
        """
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


class MeanAbsolutePercentageError(EvalMetric):
    """
    Mean Absolute Percentage Error (MAPE) evaluation metric.
    """

    name: str = 'mape'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Mean Absolute Percentage Error (MAPE).

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The MAPE value as a percentage.
        """
        return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100


class MeanAccuracy(EvalMetric):
    """
    Mean Accuracy evaluation metric.
    """

    name: str = 'accuracy'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Mean Accuracy.

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The accuracy value.
        """
        correct = torch.sum(torch.argmax(y_pred, dim=1) == y_true.squeeze())
        return (correct.item() / y_true.size(0)) * 100


class MeanCrossEntropy(EvalMetric):
    """
    Mean Cross Entropy evaluation metric.
    """

    name: str = 'cross_entropy'

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the Mean Cross Entropy.

        Args:
            y_true (Tensor): The ground truth values.
            y_pred (Tensor): The predicted values.

        Returns:
            float: The cross entropy value.
        """
        return torch.nn.functional.cross_entropy(y_pred, y_true, reduction='mean').item()
