import os, glob
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from datasetformat import datasetformat
from cnn1d import cnn1d

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = get_device()
    print("Using device:", device)

    # Quick sanity check: how many files?
    n_files = len(glob.glob("data/train/*.npz"))
    print("Found .npz files:", n_files)
    if n_files == 0:
        raise FileNotFoundError("No .npz files in data/train. Put your dataset there (audio, speaker_xy, room_rect).")

    ds = datasetformat("data/train")

    # Pick a batch size that won't zero out your DataLoader
    batch_size = min(32, len(ds))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    net = cnn1d().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    crit = nn.SmoothL1Loss()

    net.train()
    for epoch in range(20):
        running = 0.0
        num_batches = 0
        for x, y_spk, y_room in dl:
            x = x.to(device).float()
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

            y_spk  = y_spk.to(device).float().view(-1, 4)
            y_room = y_room.to(device).float()

            out = net(x)
            loss = crit(out["spk"], y_spk) + 0.5 * crit(out["room"], y_room)
            opt.zero_grad(); loss.backward(); opt.step()

            running += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("DataLoader produced 0 batches. Reduce batch_size or add more data.")
        print(f"epoch {epoch+1:02d}  mean_loss {running/num_batches:.4f}")

if __name__ == "__main__":
    main()
