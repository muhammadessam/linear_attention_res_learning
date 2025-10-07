from torch.utils.data import Dataset
import numpy as np
import os
import torch
from typing import Tuple


class NPYDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        audio_transform=None,
        video_transform=None,
    ):
        assert split in {"train", "val", "test"}
        self.root = root
        self.split = split

        self.audio = np.load(os.path.join(root, f"audio_{split}.npy"))
        self.video = np.load(os.path.join(root, f"videos_{split}.npy"))
        self.labels = np.load(os.path.join(root, f"labels_{split}.npy"))

        assert len(self.audio) == len(self.video) == len(self.labels)
        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.audio[idx]
        v = self.video[idx]
        y = int(self.labels[idx])

        # Ensure (T,C,H,W)
        if a.ndim == 3:
            a = a[:, None, :, :]
        if v.ndim == 3:
            v = v[:, None, :, :]

        # Copy to contiguous tensors for PyTorch
        a = torch.from_numpy(np.array(a, copy=True)).contiguous().float()
        v = torch.from_numpy(np.array(v, copy=True)).contiguous().float()

        # Apply optional augmentations (per-sample, consistent across T)
        if self.audio_transform is not None:
            a = self.audio_transform(a)
        if self.video_transform is not None:
            v = self.video_transform(v)

        y = torch.tensor(y, dtype=torch.long)

        return a, v, y
