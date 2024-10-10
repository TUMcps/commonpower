from __future__ import annotations

import os

import torch
from torch.nn import (
    LSTM,
    Dropout,
    Flatten,
    Linear,
    Module,
    ModuleList,
    ReLU,
    TransformerEncoder,
    TransformerEncoderLayer,
    Unflatten,
)


class NNModule(Module):
    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> NNModule:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        checkpoint = torch.load(checkpoint_path)
        model = cls(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def export_to_onnx(self, path: str):
        onnx_program = torch.onnx.dynamo_export(self, torch.randn(*self.input_shape()))
        onnx_program.save(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def input_shape(self) -> list[int]:
        raise NotImplementedError

    @property
    def output_shape(self) -> list[int]:
        raise NotImplementedError


class SimpleMLP(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        n_ahead: int = 1,
        hidden_dims: list[int] = [64, 64],
        dropout_p: float = 0.5,
    ):

        super().__init__()

        self.n_input_steps = n_lookback + 1  # adding 1 for the current time step
        self.n_ahead = n_ahead

        self.n_features = n_features
        self.n_targets = n_targets

        input_dim = n_features * self.n_input_steps
        output_dim = n_targets * n_ahead

        self.layers = ModuleList()

        self.layers.append(
            Flatten()
        )  # Flatten the input from (n_batches, n_input_steps, n_features) to (n_batches, n_input_steps * n_features)

        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(Linear(prev_dim, dim))
            self.layers.append(ReLU())
            self.layers.append(Dropout(dropout_p))
            prev_dim = dim
        self.layers.append(Linear(prev_dim, output_dim))
        self.layers.append(Unflatten(1, (n_ahead, n_targets)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


class SimpleTransformer(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        n_heads: int = 1,
        n_transformer_layers: int = 6,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1  # adding 1 for the current time step
        self.n_ahead = 1

        self.n_features = n_features
        self.n_targets = n_targets

        self.layers = ModuleList()
        transformer_layer = TransformerEncoderLayer(
            d_model=n_features, nhead=n_heads, dropout=dropout_p, batch_first=True
        )
        self.layers.append(TransformerEncoder(transformer_layer, num_layers=n_transformer_layers))  # encoder
        self.layers.append(Linear(n_features, n_targets))  # decoder

    def forward(self, x):

        x = self.layers[0](x)  # transformer
        x = self.layers[1](x[:, -1, :].unsqueeze(1))  # linear of last element in sequence
        return x

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


class SimpleLSTM(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1  # adding 1 for the current time step
        self.n_ahead = 1

        self.n_features = n_features
        self.n_targets = n_targets

        self.layers = ModuleList()
        self.layers.append(
            LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_p,
            )
        )
        self.layers.append(Linear(hidden_size, n_targets))

    def forward(self, x):

        x, _ = self.layers[0](x)  # lstm
        x = self.layers[1](x[:, -1, :].unsqueeze(1))  # linear of last element in sequence
        return x

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]
