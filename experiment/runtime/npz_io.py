"""Current NPZ shard reader."""
import glob
from typing import Dict, List
import numpy as np

def _collect_npz_files(npz_dir_or_files) -> List[str]:
    if isinstance(npz_dir_or_files, (list, tuple)):
        files = list(npz_dir_or_files)
    else:
        dir_path = str(npz_dir_or_files).rstrip("/\\")
        files = sorted(glob.glob(f"{dir_path}/*.npz"))
    if not files:
        raise FileNotFoundError(f"未找到任何 .npz 文件: {npz_dir_or_files}")
    return files


def load_npz_shards(npz_dir_or_files) -> Dict[str, np.ndarray]:
    """加载分片数据。X: (N, M*n_rx)，x/y: (N,) 为 EVE 坐标标签。"""
    files = _collect_npz_files(npz_dir_or_files)
    Xs, xs, ys = [], [], []
    for f in files:
        d = np.load(f)
        for key in ("X", "x", "y"):
            if key not in d:
                raise KeyError(f"{f} 缺少必要键 {key}")
        Xs.append(d["X"])
        xs.append(d["x"])
        ys.append(d["y"])
    return {
        "X": np.vstack(Xs),
        "x": np.concatenate(xs).astype(np.float32, copy=False),
        "y": np.concatenate(ys).astype(np.float32, copy=False),
    }


