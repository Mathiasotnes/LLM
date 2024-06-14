"""
train.py
File to start training run for the model
"""

import torch
import time
import math
import numpy as np
import os
import pickle
from itertools import islice
from contextlib import nullcontext
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from data import get_tokenized_dataloader
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# -----------------------        Configuration          -----------------------
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Data
data_dir = './data/shakespeare/dataset'

# Model
block_size = 1024
vocab_size = 50304
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# Device
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
assert device is not None
device = torch.device('cpu')

# Training
EPOCHS = 5
compile_model = False
loss_fn = torch.nn.CrossEntropyLoss()


# -----------------------------------------------------------------------------
# -----------------------       Initialization          -----------------------
# -----------------------------------------------------------------------------

print(f"Running on device: {device}")

train_loader = get_tokenized_dataloader(data_dir, 'train', block_size, batch_size=1)
val_loader = get_tokenized_dataloader(data_dir, 'val', block_size, batch_size=1)

print(f"train has {len(train_loader.dataset)*block_size:,} tokens")
print(f"val has {len(val_loader.dataset)*block_size:,} tokens")

model_args = {
    'block_size': block_size,
    'vocab_size': vocab_size,
    'n_layer': n_layer,
    'n_head': n_head,
    'n_embd': n_embd,
    'dropout': dropout,
    'bias': bias,
}

model_config = GPTConfig(**model_args)
model = GPT(model_config)
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(beta1, beta2), device_type=device)

model.to(device)

# compile the model
if compile_model:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

# -----------------------------------------------------------------------------
# -----------------------         Training             ------------------------
# -----------------------------------------------------------------------------

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    train_loader_iter = iter(train_loader)
    print(f"Training loader length: {len(train_loader)}")

    i = 0
    while i < len(train_loader):
        data = next(train_loader_iter)

        # Every data instance is an input + label pair
        inputs, labels = data
        start_time = time.time()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, loss = model(inputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        last_loss = loss.item()
        tokens_per_sec = block_size * batch_size / (time.time() - start_time)
        print(f'  batch {i + 1} | loss: {last_loss:.4f} | tok/s {tokens_per_sec:.0f}')
        tb_x = epoch_index * len(train_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        i += 1

    return last_loss

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs, vloss = model(vinputs)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
