import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data configuration
data_dir = "data.txt"

# Training hyperparameters
train_batch_size = 16
eval_batch_size = 8
context_length = 256
train_split = 0.8

# Training configuration
lr = 1e-3
epochs = 5000
eval_steps = 1000
