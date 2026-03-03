import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GPT(nn.Module):
    """Simple GPT model with token embeddings."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
    
    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits = self.wte(inputs)
        loss = None
        if targets is not None:
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, inputs: torch.Tensor, max_new_tokens: int):
        """Generate new tokens given an input sequence."""
        """The model output will be stored along with the initial input sequence."""
        for _ in range(max_new_tokens):
            logits, _ = self(inputs)  

            # for all the batches, get the embeds for the last predicted sequence
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=1)       

            # get the next probable token based on the input probs
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            inputs = torch.cat([inputs, idx_next], dim=1)
        return inputs
