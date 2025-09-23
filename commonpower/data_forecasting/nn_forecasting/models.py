from __future__ import annotations

import os

import torch
from torch.nn import (
    LSTM,
    AvgPool1d,
    Conv1d,
    Dropout,
    Flatten,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    TransformerEncoder,
    TransformerEncoderLayer,
    Unflatten,
)
from torch.nn.utils import weight_norm


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


# Basic Models #


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


# Attention Models #


class AttentionLSTM(NNModule):
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

        self.lstm = LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
        )

        self.attention = Linear(hidden_size, 1)

        self.linear = Linear(hidden_size, self.n_ahead * self.n_targets)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        """Bahdanau-style additive attention"""
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=-1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        output = self.linear(context_vector).unsqueeze(1)
        return output

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


class ReverseAttentionLSTM(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout_p: float = 0.5,
        n_ahead: int = 1,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1
        self.n_ahead = n_ahead
        self.n_features = n_features
        self.n_targets = n_targets

        # Attention over input features across time
        self.attention = Linear(n_features, 1)

        # LSTM after attention-weighted input
        self.lstm = LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
        )

        # Output projection
        self.linear = Linear(hidden_size, n_ahead * n_targets)

    def forward(self, x):
        # x shape: (batch_size, time_steps, n_features)

        # Attention over time
        scores = self.attention(x).squeeze(-1)  # shape: (batch, time)
        weights = torch.softmax(scores, dim=1)  # softmax over time steps

        # Apply attention to inputs
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, n_features)

        # Expand context back to sequence: repeat for each timestep
        context_seq = context.unsqueeze(1).repeat(1, self.n_input_steps, 1)

        # Feed through LSTM
        lstm_out, _ = self.lstm(context_seq)

        # Predict from last LSTM output
        output = self.linear(lstm_out[:, -1, :]).view(-1, self.n_ahead, self.n_targets)
        return output

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


# TCN Models #


class Chomp1d(Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(dropout)

        self.conv2 = weight_norm(
            Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(dropout)

        self.net = Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)  # reverse change
        return x


class SimpleTCN(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        hidden_size: int = 256,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1  # adding 1 for the current time step
        self.n_ahead = 1

        self.n_features = n_features
        self.n_targets = n_targets

        num_channels = [hidden_size]

        for i in range(n_layers - 1):
            hidden_size = hidden_size // 2
            num_channels.append(hidden_size)

        self.layers = ModuleList()
        self.layers.append(
            TemporalConvNet(n_features, num_channels, kernel_size, dropout_p),
        )
        self.layers.append(Linear(num_channels[-1], self.n_targets))

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x[:, -1, :].unsqueeze(1))  # linear of last element in sequence
        return x

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


# Sequence to Sequence Models #


class Seq2SeqLSTM(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        n_ahead: int,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1
        self.n_ahead = n_ahead
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0.0,
        )

        self.decoder_lstm = LSTM(
            input_size=n_targets,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0.0,
        )

        self.output_layer = Linear(hidden_size, n_targets)

    def forward(self, x, decoder_input=None, teacher_forcing: bool = False):
        """
        x: (batch_size, input_seq_len, n_features)
        decoder_input: (batch_size, n_ahead, n_targets) -> only used during training with teacher forcing
        """
        batch_size = x.size(0)

        # Encode input sequence
        _, (hidden, cell) = self.encoder(x)

        # Initialize decoder input: first input is usually zeros
        decoder_input_t = torch.zeros((batch_size, 1, self.n_targets), device=x.device)

        outputs = []

        for t in range(self.n_ahead):
            out, (hidden, cell) = self.decoder_lstm(decoder_input_t, (hidden, cell))
            pred = self.output_layer(out)  # (batch_size, 1, n_targets)
            outputs.append(pred)

            if teacher_forcing and decoder_input is not None:
                # Use true value as next input
                decoder_input_t = decoder_input[:, t : t + 1, :]
            else:
                # Use own prediction as next input
                decoder_input_t = pred

        outputs = torch.cat(outputs, dim=1)  # (batch_size, n_ahead, n_targets)
        return outputs

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


class Seq2SeqTCN(NNModule):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        n_ahead: int = 33,
        hidden_size: int = 256,
        kernel_size: int = 3,
        n_layers: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1
        self.n_ahead = n_ahead
        self.n_features = n_features
        self.n_targets = n_targets

        # TCN channel configuration
        num_channels = [hidden_size] * n_layers

        self.tcn = TemporalConvNet(
            num_inputs=n_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout_p,
        )

        # Project from hidden_size to a sequence of n_ahead * n_targets
        self.linear = Linear(hidden_size, n_targets * n_ahead)

    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        tcn_out = self.tcn(x)  # (batch_size, seq_len, hidden_size)
        last_step = tcn_out[:, -1, :]  # use the final time step
        out = self.linear(last_step)  # (batch_size, n_targets * n_ahead)
        return out.view(-1, self.n_ahead, self.n_targets)

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]


# DLinear Model #


class moving_avg(Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(NNModule):
    """
    Based from Zeng, Ailing et al.
    “Are Transformers Effective for Time Series Forecasting?” AAAI Conference on Artificial Intelligence (2022).
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_lookback: int,
        n_ahead: int = 1,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()

        self.n_input_steps = n_lookback + 1
        self.n_ahead = n_ahead

        self.n_features = n_features
        self.n_targets = n_targets
        self.individual = individual
        self.kernel_size = kernel_size

        self.decomposition = series_decomp(kernel_size=self.kernel_size)

        if self.individual:
            self.linear_seasonal = ModuleList([Linear(self.n_input_steps, self.n_ahead) for _ in range(n_features)])
            self.linear_trend = ModuleList([Linear(self.n_input_steps, self.n_ahead) for _ in range(n_features)])
        else:
            self.linear_seasonal = Linear(self.n_input_steps, self.n_ahead)
            self.linear_trend = Linear(self.n_input_steps, self.n_ahead)

    def forward(self, x):
        # x: [B, T, C] = [batch, input_steps, features]
        seasonal, trend = self.decomposition(x)  # both [B, T, C]

        # Permute to [B, C, T] for linear layer if needed
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.stack(
                [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(self.n_features)], dim=1
            )
            trend_out = torch.stack([self.linear_trend[i](trend[:, i, :]) for i in range(self.n_features)], dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        output = seasonal_out + trend_out  # shape [B, C, n_ahead]
        output = output.permute(0, 2, 1)  # [B, n_ahead, C]

        # Reduce features if n_targets < n_features (optional, for multi-var to uni-var)
        if self.n_targets < self.n_features:
            output = output[:, :, : self.n_targets]

        return output

    @property
    def input_shape(self) -> list[int]:
        return [self.n_input_steps, self.n_features]

    @property
    def output_shape(self) -> list[int]:
        return [self.n_ahead, self.n_targets]
