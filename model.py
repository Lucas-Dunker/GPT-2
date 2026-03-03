import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import context_length
from tokenizer import CharTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model) -> None:
        super().__init__()
        # Create a matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)
        
        # Create a vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector with the divisor terms based on the dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model, cosine has one fewer column than sine
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        pe = pe.unsqueeze(0)  # Shape: (1, context_length, d_model)
        
        # Register pe as a buffer, so it is not considered a parameter but is part of the module's state
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the positional encodings to the input embeddings
        return x + self.pe[:,:x.size(1), :] 


class GPT(nn.Module):
    """Simple GPT model with token embeddings."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = PositionalEncoding(context_length=context_length, d_model=d_model)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits = self.wte(inputs)
        logits = self.wpe(logits)
        logits = self.fcn(logits)
        loss = None
        if targets is not None:
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, inputs: torch.Tensor, max_new_tokens: int, tokenizer: CharTokenizer):
        """Generate new tokens given an input sequence."""
        """The model output will be stored along with the initial input sequence."""
        output = inputs.clone()
        for _ in range(max_new_tokens):
            current_seq_length = inputs.size(1)

            # Truncate inputs if it exceeds context_length
            if current_seq_length > context_length:
                inputs = inputs[:, -context_length:]

            logits, _ = self(inputs)

            # For all the batches, get the logits for the last predicted token
            logits = logits[:, -1, :]  # Get the logits for the last token
            probs = F.softmax(logits, dim=1)

            # Get the probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to both inputs (for next iteration) and output
            inputs = torch.cat([inputs, idx_next], dim=1)
            output = torch.cat([output, idx_next], dim=1)
        return [tokenizer.decode(out.tolist()) for out in output]

