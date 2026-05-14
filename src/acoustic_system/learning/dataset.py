"""PyTorch Dataset wrapping the active-sensing HDF5 archive.

Reads the ``(sensor, source, obstacles)`` triplets written by
``scripts/generate_active_sensing.py`` and yields tensors ready for
the ``DualInputCNN``:

- ``sensor``: ``(n_mics, T_rec)`` float32 — channel-first (PyTorch
  convention), transposed from the HDF5's channel-last layout.
- ``source``: ``(1, T_audio)`` float32.
- ``mask``: ``(H_target, W_target)`` float32 in {0, 1}, optionally
  resized from the native obstacle resolution via nearest-neighbour
  interpolation so the model output head can be a fixed size
  irrespective of the dataset's grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ActiveSensingDataset(Dataset):
    """HDF5-backed dataset for the active-sensing CNN.

    Parameters
    ----------
    hdf5_path
        Path to an archive produced by ``scripts/generate_active_sensing.py``.
    target_mask_size
        If set, the obstacle mask is resized (nearest-neighbour) to this
        square shape before being yielded. Use ``None`` to keep the
        native shape; in that case all samples in the dataset must have
        identical mask shape (typical, since the script writes a fixed
        ``--grid``-by-``--grid`` mask per run).
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        target_mask_size: Optional[int] = 64,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.target_mask_size = target_mask_size

        # Inventory the archive in __init__ so __len__ is cheap and the
        # sample order is deterministic. We do NOT cache an open file
        # handle — h5py.File is not picklable, which would break the
        # DataLoader's multi-worker mode. Each __getitem__ opens the
        # file lazily (cheap with HDF5's memory-mapped backend).
        with h5py.File(self.hdf5_path, "r") as f:
            self.sample_keys: list[str] = sorted(k for k in f.keys() if k.startswith("sample_"))
            if not self.sample_keys:
                raise ValueError(f"no samples in {self.hdf5_path}")
            # Probe the first sample to record per-dataset metadata.
            grp = f[self.sample_keys[0]]
            self.n_mics: int = int(grp.attrs.get("n_mics", 1))
            self.t_rec: int = int(np.asarray(grp["sensor"]).shape[0])
            self.t_audio: int = int(np.asarray(grp["source"]).shape[0])
            self.native_mask_shape: Tuple[int, int] = tuple(np.asarray(grp["obstacles"]).shape)  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.sample_keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[self.sample_keys[index]]
            sensor_np = np.asarray(grp["sensor"], dtype=np.float32)  # (T_rec, n_mics)
            source_np = np.asarray(grp["source"], dtype=np.float32)  # (T_audio,)
            mask_np = np.asarray(grp["obstacles"], dtype=np.float32)  # (H, W)

        # Channel-first for PyTorch convolutions.
        sensor = torch.from_numpy(sensor_np.T.copy())  # (n_mics, T_rec)
        source = torch.from_numpy(source_np.copy()).unsqueeze(0)  # (1, T_audio)
        mask = torch.from_numpy(mask_np.copy())  # (H, W)

        if self.target_mask_size is not None and mask.shape != (
            self.target_mask_size,
            self.target_mask_size,
        ):
            # Nearest-neighbour interpolation preserves the binary nature
            # of the mask. F.interpolate wants (N, C, H, W).
            mask = torch.nn.functional.interpolate(
                mask[None, None],
                size=(self.target_mask_size, self.target_mask_size),
                mode="nearest",
            )[0, 0]

        return sensor, source, mask
