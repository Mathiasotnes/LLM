"""
data/dataloader.py
Dataloader for tokenized dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.block_size = block_size
        self.data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.fromfile(self.data_path, dtype=np.uint16)
        print(f"loaded {len(self.data):,} tokens from {self.data_path}")

    def __len__(self):
        # Number of blocks
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        x = torch.from_numpy(self.data[start_idx:start_idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[start_idx + 1:start_idx + 1 + self.block_size].astype(np.int64))
        return x, y

def get_tokenized_dataloader(data_dir, split, block_size, batch_size, shuffle=False):
    dataset = TokenizedDataset(data_dir, split, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader