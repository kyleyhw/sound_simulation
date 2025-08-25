import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class datasetformat(Dataset):
    """
    Expecting each .npz to contain:
      - audio: float32 array [2, T] (channel 0 / 1 = left / right speaker time series)
      - speaker_xy: int/float array [2, 2]  [[xL,yL],[xR,yR]] in grid units
      - room_rect: float array [4]  [xmin, xmax, ymin, ymax] (grid units)
    """
    def __init__(self, root):
        self.paths = sorted(glob.glob(f"{root}/*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        z = np.load(self.paths[i], allow_pickle=False)
        x = z["audio"].astype(np.float32)            # [2, T]

        spk = z["speaker_xy"].astype(np.float32)     # [2,2] -> (xL,yL,xR,yR)
        y_spk = spk.reshape(-1)                       # [4]

        xmin, xmax, ymin, ymax = z["room_rect"].astype(np.float32)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        w,  h  = (xmax - xmin), (ymax - ymin)
        y_room = np.array([cx, cy, w, h], dtype=np.float32)  # [4]

        return torch.from_numpy(x), torch.from_numpy(y_spk), torch.from_numpy(y_room)
