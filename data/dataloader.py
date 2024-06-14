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
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y

def get_tokenized_dataloader(data_dir, split, block_size, batch_size, shuffle=False):
    dataset = TokenizedDataset(data_dir, split, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader