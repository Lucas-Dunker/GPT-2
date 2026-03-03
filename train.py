import torch

from config import (
    device, data_dir, train_batch_size, eval_batch_size,
    context_length, train_split, lr, epochs, eval_steps
)
from tokenizer import CharTokenizer
from dataloader import DataLoader
from model import GPT


def main():
    text = open(data_dir, 'r').read()
    
    # Tokenizer setup
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    
    # Split data into train and eval
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]
    
    # Data Loader setup
    train_loader = DataLoader(train_data, train_batch_size, context_length)
    eval_loader = DataLoader(eval_data, eval_batch_size, context_length)
    
    # Model setup
    d_model = vocab_size
    model = GPT(vocab_size=vocab_size, d_model=d_model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for ep in range(epochs):
        xb, yb = train_loader.get_batch()
        
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        
        if ep % eval_steps == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                xvb, yvb = eval_loader.get_batch()
                _, e_loss = model(xvb, yvb)
                print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss}\teval_loss: {e_loss}")
            model.train()
    
    # Generate sample text
    with torch.no_grad():
        input_tensor = torch.tensor(
            tokenizer.encode("Love"), dtype=torch.long, device=device
        ).unsqueeze(0)
        output = model.generate(input_tensor, max_new_tokens=500, tokenizer=tokenizer)[0]
        print(output)


if __name__ == "__main__":
    main()
