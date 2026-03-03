import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from config import context_length
from tokenizer import CharTokenizer

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape
        
        # Project the input embeddings into Q, K, and V
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        
        # Apply mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        out = self.fc_out(attention_output)
        
        return out

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
        self.att = SelfAttention(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(d_model, vocab_size)
    
    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        logits = self.wte(inputs)
        logits = self.wpe(logits)

        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)

        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        logits = self.linear1(logits)
        

        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
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

