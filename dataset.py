import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96))
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx], 0)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask).long()

        return img, mask
