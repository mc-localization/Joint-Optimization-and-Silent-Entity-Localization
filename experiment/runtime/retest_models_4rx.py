"""公平重测用的定位模型。

所有候选模型共享：
1. 非负计数上的 log1p；
2. 输入 BatchNorm；
3. 统一 Xavier 初始化；
4. 输出为标准化坐标 (x, y)。

GRU is provided as a temporal alternative to CNN. A small Transformer is also
provided as an optional ablation. Vanilla RNN is intentionally not used as the
main replacement because its long-window optimization is less stable.

候选配置只改变容量、深度和 dropout，避免把预处理差异误判成模型差异。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def _init_xavier(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.GRU):
            for name, parameter in layer.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        elif isinstance(layer, nn.MultiheadAttention):
            if layer.in_proj_weight is not None:
                nn.init.xavier_uniform_(layer.in_proj_weight)
            if layer.in_proj_bias is not None:
                nn.init.zeros_(layer.in_proj_bias)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def _preprocess(x: torch.Tensor, input_bn: nn.BatchNorm1d) -> torch.Tensor:
    # Molecular counts are non-negative. Clamping protects the transform if a
    # corrupted shard contains a small negative value.
    x = torch.log1p(torch.clamp(x, min=0.0))
    return input_bn(x)


class CNNLocalizerV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int = 48,
        pooled_bins: int = 4,
        mlp_hidden: int = 128,
        dropout: float = 0.05,
        n_rx: int = 2,
    ) -> None:
        super().__init__()
        if input_dim % n_rx != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by n_rx={n_rx}")
        if pooled_bins < 1:
            raise ValueError("pooled_bins must be positive")
        self.n_rx = int(n_rx)
        self.seq_len = input_dim // self.n_rx
        self.pooled_bins = int(pooled_bins)
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(self.n_rx, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        # Parallel receptive fields are useful when L is short, while the
        # concatenation keeps local and medium-scale slopes separate.
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, k, padding=k // 2),
                          nn.BatchNorm1d(channels), nn.ReLU())
            for k in (3, 5, 9)
        ])
        mixed = channels * len(self.branches)
        self.mix = nn.Sequential(
            nn.Conv1d(mixed, channels * 2, kernel_size=1),
            nn.BatchNorm1d(channels * 2), nn.ReLU(),
        )
        width = channels * 2
        self.res1 = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width),
        )
        self.res_act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(self.pooled_bins)
        self.head = nn.Sequential(
            nn.Linear(width * self.pooled_bins, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, 2),
        )
        _init_xavier(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _preprocess(x, self.input_bn)
        x = x.view(x.size(0), self.n_rx, self.seq_len)
        x = self.stem(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.mix(x)
        x = self.res_act(x + self.res1(x))
        # Keep coarse absolute time positions instead of global averaging.
        x = self.pool(x).flatten(1)
        return self.head(x)


class GRULocalizerV2(nn.Module):
    """Temporal localizer with a parameter budget close to the matched set.

    The input is reshaped from [receiver_1 sequence, receiver_2 sequence] to
    [time, receiver_features], so each GRU step observes both receivers at the
    same time index. The default 2x96 GRU plus a 192-dimensional head is about
    100k parameters at l=50.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        head_hidden: int = 192,
        dropout: float = 0.05,
        n_rx: int = 2,
    ) -> None:
        super().__init__()
        if input_dim % n_rx != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by n_rx={n_rx}")
        if num_layers not in (1, 2, 3):
            raise ValueError("num_layers must be 1, 2, or 3")
        self.n_rx = int(n_rx)
        self.seq_len = input_dim // self.n_rx
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=self.n_rx,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 2),
        )
        _init_xavier(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _preprocess(x, self.input_bn)
        x = x.view(x.size(0), self.n_rx, self.seq_len).transpose(1, 2).contiguous()
        sequence, _ = self.gru(x)
        # Use the complete temporal window instead of relying only on the
        # final hidden state. This is better suited to cumulative absorption
        # curves, where informative changes may occur before the last sample.
        return self.head(sequence.mean(dim=1))


def _sinusoidal_position_encoding(
    length: int, dimension: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.unsqueeze(0)


class TransformerLocalizerV2(nn.Module):
    """Small attention-based localizer for an optional ablation.

    Sinusoidal positions avoid adding a length-dependent learned embedding.
    The default d_model=64, two encoder layers, and feed-forward width 192 are
    deliberately small so the model remains close to the 100k-parameter set.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        head_hidden: int = 192,
        dropout: float = 0.05,
        n_rx: int = 2,
    ) -> None:
        super().__init__()
        if input_dim % n_rx != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by n_rx={n_rx}")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.n_rx = int(n_rx)
        self.seq_len = input_dim // self.n_rx
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(self.n_rx, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 2),
        )
        _init_xavier(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _preprocess(x, self.input_bn)
        x = x.view(x.size(0), self.n_rx, self.seq_len).transpose(1, 2).contiguous()
        x = self.input_proj(x)
        x = x + _sinusoidal_position_encoding(
            self.seq_len, x.size(-1), x.device, x.dtype
        )
        x = self.encoder(x)
        weights = torch.softmax(self.pool_score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)


class DNNLocalizerV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if num_hidden_layers not in (2, 3, 4, 5):
            raise ValueError("num_hidden_layers must be between 2 and 5")
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)
        _init_xavier(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(_preprocess(x, self.input_bn)))






MODEL_CLASSES = {
    "cnn": CNNLocalizerV2,
    "gru": GRULocalizerV2,
    "transformer": TransformerLocalizerV2,
    "dnn": DNNLocalizerV2,
}


def build_model(name: str, input_dim: int, cfg: Mapping[str, Any], n_rx: int = 2) -> nn.Module:
    if name not in MODEL_CLASSES:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_CLASSES)}")
    params = dict(cfg)
    if name in ("cnn", "gru", "transformer"):
        params["n_rx"] = n_rx
    return MODEL_CLASSES[name](input_dim=input_dim, **params)
