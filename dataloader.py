import torch


class DataLoader:
    """DataLoader for batching tokenized text data."""
    
    def __init__(self, tokens: torch.Tensor, batch_size: int, context_length: int) -> None:
        self.tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.current_position = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of input-target pairs."""
        b, c = self.batch_size, self.context_length
        
        # Randomly sample starting positions for each sequence in the batch
        ix = torch.randint(0, len(self.tokens) - c, (b,))
        x = torch.stack([self.tokens[i:i+c] for i in ix])
        y = torch.stack([self.tokens[i+1:i+c+1] for i in ix])
        return x, y
