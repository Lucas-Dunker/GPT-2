import torch

from config import (
    device, data_dir, train_batch_size, eval_batch_size,
    context_length, train_split, lr, epochs, eval_steps, num_heads,
    n_layers
)
import tiktoken
from dataloader import DataLoader
from model import GPT



def main():
    text = open(data_dir, 'r').read()
    
    # Tokenizer setup
    tokenizer = tiktoken.get_encoding('gpt2')
    vocab_size = tokenizer.n_vocab
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    
    # Split data into train and eval
    n_data = len(data)
    train_data = data[:int(n_data * train_split)]
    eval_data = data[int(n_data * train_split):]
    
    # Data Loader setup
    train_loader = DataLoader(train_data, train_batch_size, context_length)
    eval_loader = DataLoader(eval_data, eval_batch_size, context_length)
    
    # Model setup
    d_model = 512
    model = GPT(vocab_size=vocab_size, d_model=d_model, n_heads=num_heads, n_layers=n_layers).to(device)
    model = torch.compile(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=3000, eta_min=lr*0.1)

    print(model)
    print(f"Number of parameters: {round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000)}M")
    
    train_loss = {}
    
    # Training loop
    for ep in range(epochs):
        xb, yb = train_loader.get_batch()
        
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optim.step()
        scheduler.step()
        train_loss[ep] = loss.item()
        
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
