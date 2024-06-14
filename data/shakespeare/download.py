"""
data/shakespeare/download.py
Download Shakespeare dataset from the internet and tokenize it, then save it to a file.
"""

import requests
import tiktoken
import numpy as np

def download_shakespeare(file_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    response = requests.get(url)
    print(f"downloaded {len(response.text):,} characters")
    with open(file_path, 'w') as f:
        f.write(response.text)
    

def tokenize(file_path, tokenizer):
    with open(file_path, 'r') as f:
        texts = f.readlines()
    tokens = []
    for text in texts:
        tokenized = tokenizer.encode(text)
        for token in tokenized:
            tokens.append(token)
    return tokens


if __name__ == '__main__':
    file_path = 'data/shakespeare/dataset/shakespeare.txt'
    download_shakespeare(file_path)
    tokenizer = tiktoken.get_encoding('gpt2')
    tokens = tokenize(file_path, tokenizer)
    tokens = np.array(tokens, dtype=np.uint16)
    train_tokens, val_tokens = np.split(tokens, [int(0.9 * len(tokens))])
    print(f"train has {len(train_tokens):,} tokens")
    print(f"val has {len(val_tokens):,} tokens")
    train_tokens.tofile('data/shakespeare/dataset/train_tok.bin')
    val_tokens.tofile('data/shakespeare/dataset/val_tok.bin')