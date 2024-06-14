# data/dataloader.py

from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class TextDataset(Dataset):
    def __init__(self, file_path, max_length=128):
        with open(file_path, 'r') as f:
            self.texts = f.readlines()
        self.tokenizer = tiktoken.Encoding('gpt2')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.max_length]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens)

def get_dataloader(file_path, max_length=128, batch_size=32, shuffle=True):
    dataset = TextDataset(file_path, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
