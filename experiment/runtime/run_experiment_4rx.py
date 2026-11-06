"""Current localizer, data and training primitives; no old experiment entry points."""

from __future__ import annotations

import argparse
import fcntl
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
JOURNAL_ROOT = HERE
sys.path.insert(0, str(JOURNAL_ROOT))
sys.path.insert(0, str(HERE))

from npz_io import load_npz_shards  # noqa: E402
from retest_models_4rx import build_model  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str, base: Path = HERE) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def get_output_dir(cfg: Mapping[str, Any]) -> Path:
    path = resolve_path(str(cfg["paths"]["output_dir"]))
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_seed(cfg: Mapping[str, Any], seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(cfg["training"].get("deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def choose_device(cfg: Mapping[str, Any]) -> torch.device:
    requested = str(cfg["training"].get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def subset(data: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in data.items()}


def sample_split(data: Mapping[str, np.ndarray], count: int, seed: int) -> dict[str, np.ndarray]:
    """Randomly sample the complete split, never a prefix slice."""
    n = int(len(data["x"]))
    if count <= 0 or count >= n:
        return {key: np.asarray(value) for key, value in data.items()}
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=count, replace=False).astype(np.int64))
    return subset(data, indices)


def resample_features(
    data: Mapping[str, np.ndarray], source_M: int, target_M: int, n_rx: int
) -> dict[str, np.ndarray]:
    """Convert stored per-Rx sequences to the model-side temporal length.

    The existing Monte Carlo files contain ``source_M`` samples per Rx.  This
    operation only changes the representation consumed by the models; it does
    not rerun the physical simulation or modify the source files.  The index
    grid preserves both endpoints of each Rx sequence.
    """
    raw_x = np.asarray(data["X"])
    expected_source = int(source_M) * int(n_rx)
    expected_target = int(target_M) * int(n_rx)
    if raw_x.ndim != 2:
        raise ValueError(f"expected feature matrix, got shape {raw_x.shape}")
    if raw_x.shape[1] == expected_target:
        return {key: np.asarray(value) for key, value in data.items()}
    if raw_x.shape[1] != expected_source:
        raise ValueError(
            f"feature width {raw_x.shape[1]} is incompatible with source_M="
            f"{source_M}, target_M={target_M}, n_rx={n_rx}"
        )
    if target_M > source_M:
        raise ValueError("target_M must not exceed source_M for downsampling")

    indices = np.rint(
        np.linspace(0, int(source_M) - 1, int(target_M), dtype=np.float64)
    ).astype(np.int64)
    pieces = [raw_x[:, rx * source_M + indices] for rx in range(n_rx)]
    converted = dict(data)
    converted["X"] = np.concatenate(pieces, axis=1)
    return converted


class TargetScaler:
    """Coordinate standardization fitted on the training labels only."""

    def __init__(self, train: Mapping[str, np.ndarray]) -> None:
        self.x_mean = float(np.mean(train["x"]))
        self.x_std = float(np.std(train["x"]) + 1e-6)
        self.y_mean = float(np.mean(train["y"]))
        self.y_std = float(np.std(train["y"]) + 1e-6)

    def normalize(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.column_stack(
            [
                ((data["x"] - self.x_mean) / self.x_std).astype(np.float32),
                ((data["y"] - self.y_mean) / self.y_std).astype(np.float32),
            ]
        )

    def physical(self, values: np.ndarray) -> np.ndarray:
        result = np.empty_like(values, dtype=np.float64)
        result[:, 0] = values[:, 0] * self.x_std + self.x_mean
        result[:, 1] = values[:, 1] * self.y_std + self.y_mean
        return result


class WindowDataset(Dataset):
    """Return a cropped, concatenated [Rx1, Rx2] window and a 2D target."""

    def __init__(
        self,
        raw_x: np.ndarray,
        targets: np.ndarray,
        length: int,
        start: int,
        M: int,
        n_rx: int,
        precompute: bool,
        input_mode: str = "cumulative",
    ) -> None:
        expected = int(M) * int(n_rx)
        if raw_x.ndim != 2 or raw_x.shape[1] != expected:
            raise ValueError(f"expected X shape (N,{expected}), got {raw_x.shape}")
        if not (1 <= length <= M):
            raise ValueError(f"length must be in [1,{M}], got {length}")
        xmax = M - length + 1
        if not (1 <= start <= xmax):
            raise ValueError(f"start must be in [1,{xmax}], got {start}")

        self.raw_x = np.asarray(raw_x)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.length = int(length)
        self.start = int(start)
        self.M = int(M)
        self.n_rx = int(n_rx)
        self.left = self.start - 1
        self.right = self.left + self.length
        self.input_mode = str(input_mode)
        if self.input_mode not in {"cumulative", "increment"}:
            raise ValueError("input_mode must be cumulative or increment")
        self.features: np.ndarray | None = None

        if precompute:
            # Allocate once; avoid full-sequence float copies and concatenate peaks.
            self.features = np.empty((len(self.raw_x), self.n_rx*self.length), dtype=np.float32)
            for rx in range(self.n_rx):
                offset = rx*self.M
                values = self.features[:, rx*self.length:(rx+1)*self.length]
                values[:] = self.raw_x[:, offset+self.left:offset+self.right]
                if self.input_mode == "increment" and self.left > 0:
                    values -= self.raw_x[:, offset+self.left-1:offset+self.left].astype(np.float32)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        if self.features is not None:
            feature = self.features[index]
        else:
            row = self.raw_x[index]
            chunks = []
            for rx in range(self.n_rx):
                sequence = row[rx * self.M:(rx + 1) * self.M].astype(np.float32, copy=False)
                window = sequence[self.left:self.right].copy()
                if self.input_mode == "increment":
                    baseline = sequence[self.left - 1] if self.left > 0 else 0.0
                    window -= baseline
                chunks.append(window)
            feature = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(feature), torch.from_numpy(self.targets[index])


def load_splits(
    cfg: Mapping[str, Any], *, final: bool, include_test: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None,
           dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_root = resolve_path(str(cfg["paths"]["data_dir"]))
    target_M = int(cfg["data"]["M"])
    source_M = int(cfg["data"].get("source_M", target_M))
    n_rx = int(cfg["data"]["n_rx"])
    train_all = resample_features(
        load_npz_shards(data_root / "data_train"), source_M, target_M, n_rx
    )
    valid_all = resample_features(
        load_npz_shards(data_root / "data_valid"), source_M, target_M, n_rx
    )
    if final:
        train_count = int(cfg["data"]["final_train_samples"])
        valid_count = int(cfg["data"]["final_valid_samples"])
    else:
        train_count = int(cfg["data"]["search_train_samples"])
        valid_count = int(cfg["data"]["search_valid_samples"])

    sample_seed = int(cfg["data"]["sample_seed"])
    train = sample_split(train_all, train_count, sample_seed)
    valid = sample_split(valid_all, valid_count, sample_seed + 1)
    test = None
    if include_test:
        test = resample_features(
            load_npz_shards(data_root / "data_test"), source_M, target_M, n_rx
        )
    return train, valid, test, train_all, valid_all


def make_dataset(
    data: Mapping[str, np.ndarray], scaler: TargetScaler, length: int, start: int,
    cfg: Mapping[str, Any]
) -> WindowDataset:
    M = int(cfg["data"]["M"])
    n_rx = int(cfg["data"]["n_rx"])
    bytes_needed = len(data["x"]) * n_rx * length * 4
    max_bytes = int(cfg["data"].get("max_precompute_bytes", 256 * 1024 * 1024))
    precompute = bool(cfg["data"].get("precompute_windows", True)) and bytes_needed <= max_bytes
    dataset = WindowDataset(
        data["X"], scaler.normalize(data), length, start, M, n_rx, precompute,
        input_mode=str(cfg["data"].get("input_mode", "cumulative")),
    )
    dataset.physical_scale = (scaler.x_std, scaler.y_std)
    return dataset


def batch_size_for(cfg: Mapping[str, Any], length: int) -> int:
    threshold = int(cfg["training"].get("long_window_threshold", 200))
    if length >= threshold:
        return int(cfg["training"].get("long_window_batch_size", cfg["training"]["batch_size"]))
    return int(cfg["training"]["batch_size"])


def make_loader(
    dataset: Dataset, cfg: Mapping[str, Any], device: torch.device,
    length: int, shuffle: bool
) -> DataLoader:
    workers = int(cfg["training"].get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size_for(cfg, length),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def model_config(cfg: Mapping[str, Any], model_name: str) -> tuple[str, dict[str, Any]]:
    candidates = cfg["models"].get(model_name)
    if not candidates:
        raise KeyError(f"missing model config for {model_name}")
    item = dict(candidates[0])
    config_id = str(item.pop("id"))
    return config_id, item


def build_localizer(cfg: Mapping[str, Any], model_name: str, length: int) -> nn.Module:
    _, params = model_config(cfg, model_name)
    return build_model(
        model_name,
        int(cfg["data"]["n_rx"]) * length,
        params,
        n_rx=int(cfg["data"]["n_rx"]),
    )


def forward_model(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    prediction = model(xb)
    if isinstance(prediction, tuple):
        prediction = torch.stack(prediction, dim=1)
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise RuntimeError(f"model output must have shape (batch,2), got {tuple(prediction.shape)}")
    return prediction


def localizer_loss(prediction, target, physical_scale, loss_name):
    if loss_name == 'mean_euclidean_um':
        residual = (prediction.float()-target.float())*physical_scale
        return torch.linalg.vector_norm(residual, ord=2, dim=1).mean()
    if loss_name == 'mse_physical_um2':
        residual = (prediction.float()-target.float())*physical_scale
        return residual.square().sum(dim=1).mean()
    if loss_name == 'mse_normalized':
        return (prediction-target).square().mean()
    raise ValueError('Unknown localizer loss: '+str(loss_name))


def train_localizer(
    cfg: Mapping[str, Any], model_name: str, length: int, train_ds: Dataset,
    valid_ds: Dataset, device: torch.device, max_epochs: int, patience: int,
) -> tuple[dict[str, torch.Tensor], float, int, float]:
    loss_name = cfg['training'].get('loss', 'mse_normalized')
    if loss_name in {'mean_euclidean_um', 'mse_physical_um2'} and not hasattr(train_ds, 'physical_scale'):
        raise ValueError('Physical loss requires a training-only coordinate scaler')
    physical_scale = torch.tensor(getattr(train_ds, 'physical_scale', (1., 1.)), dtype=torch.float32, device=device)
    model = build_localizer(cfg, model_name, length).to(device)
    train_loader = make_loader(train_ds, cfg, device, length, shuffle=True)
    valid_loader = make_loader(valid_ds, cfg, device, length, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_sum=0.0
        train_count=0
        model.train()
        for xb, target in train_loader:
            xb = xb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = localizer_loss(forward_model(model, xb), target, physical_scale, loss_name)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            train_sum+=float(loss.detach())*int(target.shape[0])
            train_count+=int(target.shape[0])

        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for xb, target in valid_loader:
                xb = xb.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    batch_loss = localizer_loss(forward_model(model, xb), target, physical_scale, loss_name)
                count = int(target.shape[0])
                total_loss += float(batch_loss.item()) * count
                total_count += count

        val_loss = total_loss / max(1, total_count)
        observer=globals().get("TRAINING_OBSERVER")
        if observer is not None: observer(epoch,train_sum/max(1,train_count),val_loss)
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    return best_state, float(best_loss), int(best_epoch), time.perf_counter() - started


def checkpoint_path(
    cfg: Mapping[str, Any], stage: str, model_name: str, length: int, start: int,
    seed: int, epochs: int, patience: int
) -> Path:
    config_id, params = model_config(cfg, model_name)
    payload = {
        "stage": stage,
        "model": model_name,
        "config_id": config_id,
        "params": params,
        "length": int(length),
        "start": int(start),
        "seed": int(seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "data": cfg["data"],
        "training": cfg["training"],
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(cfg["paths"].get("checkpoint_cache_dir", str(get_output_dir(cfg)))) / "checkpoints" / (
        f"{stage}_{model_name}_l{length}_s{start}_seed{seed}_{digest}.pt"
    )


def train_or_load(cfg, stage, model_name, length, start, seed, train_ds, valid_ds, device, epochs, patience):
    path = checkpoint_path(cfg, stage, model_name, length, start, seed, epochs, patience)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _train_or_load_unlocked(cfg, stage, model_name, length, start, seed,
                                      train_ds, valid_ds, device, epochs, patience)


def _train_or_load_unlocked(
    cfg: Mapping[str, Any], stage: str, model_name: str, length: int, start: int,
    seed: int, train_ds: Dataset, valid_ds: Dataset, device: torch.device,
    epochs: int, patience: int,
) -> tuple[dict[str, torch.Tensor], float, int, float, bool]:
    path = checkpoint_path(cfg, stage, model_name, length, start, seed, epochs, patience)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["state"], float(payload["val_loss"]), int(payload["best_epoch"]), 0.0, True

    configure_seed(cfg, seed)
    state, val_loss, best_epoch, elapsed = train_localizer(
        cfg, model_name, length, train_ds, valid_ds, device, epochs, patience
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(
        {
            "state": state,
            "val_loss": val_loss,
            "loss_name": cfg["training"].get("loss", "mse_normalized"),
            "best_epoch": best_epoch,
            "seed": seed,
        },
        temporary,
    )
    temporary.replace(path)
    return state, val_loss, best_epoch, elapsed, False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)




def propose_candidate_groups(
    rng: np.random.Generator,
    x0: int,
    length: int,
    M: int,
    k_perturb: int,
    k_local: int,
    k_random: int,
    iter_idx: int,
    perturb_scale: float,
    decay: float,
    local_radius: int,
) -> list[tuple[str, int]]:
    """Generate the three candidate groups used by the journal-code search."""
    if length == M:
        return [("boundary", 1)]
    xmin, xmax = 1, M - length + 1
    x0 = max(xmin, min(xmax, int(x0)))
    scale = max(1, int(round(perturb_scale * (decay ** iter_idx))))
    selected: dict[int, str] = {}

    def add(phase: str, value: int) -> None:
        value = max(xmin, min(xmax, int(value)))
        selected.setdefault(value, phase)

    for _ in range(k_perturb):
        add("perturb", x0 + int(rng.integers(-scale, scale + 1)))
    left = max(xmin, x0 - int(local_radius))
    right = min(xmax, x0 + int(local_radius))
    for _ in range(k_local):
        add("local", int(rng.integers(left, right + 1)))
    for _ in range(k_random):
        add("random", int(rng.integers(xmin, xmax + 1)))
    if not selected:
        add("random", int(rng.integers(xmin, xmax + 1)))
    return [(phase, start) for start, phase in sorted(selected.items())]




class PolicyNet(nn.Module):
    def __init__(self, hidden_dims: Iterable[int] = (32, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for width in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(width)), nn.ReLU()])
            in_dim = int(width)
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def decode_start(fraction: float, length: int, M: int) -> int:
    xmax = M - length + 1
    return int(round(1 + float(fraction) * max(0, xmax - 1)))










@torch.no_grad()
def physical_metrics(
    cfg: Mapping[str, Any], state: Mapping[str, torch.Tensor], model_name: str,
    length: int, dataset: Dataset, raw: Mapping[str, np.ndarray], scaler: TargetScaler,
    device: torch.device,
) -> dict[str, float]:
    model = build_localizer(cfg, model_name, length).to(device)
    model.load_state_dict(state)
    model.eval()
    values: list[np.ndarray] = []
    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    for xb, _ in make_loader(dataset, cfg, device, length, shuffle=False):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            values.append(forward_model(model, xb.to(device, non_blocking=True)).float().cpu().numpy())
    predicted = scaler.physical(np.concatenate(values, axis=0))
    truth = np.column_stack([raw["x"], raw["y"]]).astype(np.float64)
    errors = np.linalg.norm(predicted - truth, axis=1)
    threshold = float(cfg["evaluation"]["success_threshold_um"])
    return {
        "mean_error_um": float(np.mean(errors)),
        "median_error_um": float(np.median(errors)),
        "p90_error_um": float(np.quantile(errors, 0.90)),
        "success_rate": float(np.mean(errors < threshold)),
    }












