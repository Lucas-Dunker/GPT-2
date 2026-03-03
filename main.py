import torch

data_dir = "data.txt"
text = open(data_dir, 'r').read() # load all the data as simple string

# Get all unique characters in the text as vocabulary
chars = list(set(text))
vocab_size = len(chars)

# build the character level tokenizer
chr_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_chr = {i:c for i, c in enumerate(chars)}

def encode(input_text: str) -> list[int]:
    return [chr_to_idx[t] for t in input_text]

def decode(input_tokens: list[int]) -> str:
    return "".join([idx_to_chr[i] for i in input_tokens])


# convert our text data into tokenized tensor
data = torch.tensor(encode(text), dtype=torch.long, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))