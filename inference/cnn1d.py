import torch, torch.nn as nn

class cnn1d(nn.Module):
    def __init__(self, T_out_room=4, T_out_spk=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            # MaxPool1d halves temporal resolution each time, eg 500 -> 250 -> 125
            # kernel size = 7, padding = 3
            # 2 input channels, learns 32 different temporal filters (output channels)
            # ReLU: activation function f(x)=max(0,x)
            # Note: if underfitting/high bias, increase output channels
            # Note: if overfitting, decrease output channels.
            nn.Conv1d(32,64,7,padding=3),   nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,7,padding=3),  nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            # AdaptiveAvgPool1d averages over remaining time axis, to produce global descriptor
            # In form [B, 128, 1]
        )
        self.head_room = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, T_out_room))
        # 4 numbers, treating room as rectangle for now. In form (cx, cy, w, h). 
        self.head_spk  = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, T_out_spk))
        # 4 numbers, 2 speaker positions. In form (xL, yL, xR, yR). 
        # Note: may need to normalise to [0,1] and bound with tanh/sigmoid to keep predictions in range.
    def forward(self, x):  
        # called from next(x) in trainingtest.py, once every batch.
        h = self.backbone(x)
        return {"room": self.head_room(h), "spk": self.head_spk(h)}
        # x is in form [B, 2, T]. 
        # T: no. of time samples, equal to duration / timestep, eg 5.0s / 0.01s = 500
        # B: batch size
        # 2: channels (speakers left and right)
