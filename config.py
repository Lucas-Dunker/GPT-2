import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data configuration
data_dir = "data.txt"

# Model configuration
context_length = 512
num_heads = 4 # attention heads, should be divisible with d_model
n_layers = 1 # number of gpt blocks/layers

# Training hyperparameters
train_batch_size = 16
eval_batch_size = 8
train_split = 0.8

# Training configuration
lr = 1e-3
epochs = 3500
eval_steps = 100
