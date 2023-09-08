import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' 
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to unique indices
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # encoder: string -> list of indices
decode = lambda x: ''.join([itos[i] for i in x]) # decoder: list of indices -> string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

