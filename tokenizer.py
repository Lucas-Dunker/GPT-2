class CharTokenizer:
    """Character-level tokenizer for our model."""
    
    def __init__(self, text: str):
        # Get all unique characters in the text as vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Build the character-level dictionary mappings
        self.chr_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_chr = {i: c for i, c in enumerate(self.chars)}
    
    def encode(self, input_text: str) -> list[int]:
        """Convert a string to a list of token indices."""
        return [self.chr_to_idx[t] for t in input_text]
    
    def decode(self, input_tokens: list[int]) -> str:
        """Convert a list of token indices back to a string."""
        return "".join([self.idx_to_chr[i] for i in input_tokens])
