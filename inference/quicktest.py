import os, numpy as np
os.makedirs("data/train", exist_ok=True)
T = 512
for i in range(16):
    audio = (np.random.randn(2, T) * 0.05).astype(np.float32)
    speaker_xy = np.array([[64, 80], [192, 80]], dtype=np.float32)
    room_rect  = np.array([16, 240, 16, 240], dtype=np.float32)  # xmin,xmax,ymin,ymax
    np.savez_compressed(f"data/train/scene_{i:05d}.npz",
                        audio=audio, speaker_xy=speaker_xy, room_rect=room_rect)
print("Wrote dummy samples to data/train")
