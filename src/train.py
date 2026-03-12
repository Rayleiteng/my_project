import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_math_dataloader
from math_embedding import MathTextbookEmbedding
from transformer_block import TransformerBlock
class MathGPT(nn.Module):
        def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4):
            super().__init__()
            self.embedding = MathTextbookEmbedding(vocab_size, d_model)
            
            self.blocks = nn.Sequential(
                *[TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
            )
            
            self.ln_f = nn.LayerNorm(d_model)
            
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits

if __name__ == "__main__":
    dataset, dataloader = get_math_dataloader(
        file_path="../data/processed_latex/train_data_100_all.mmd",
        tokenizer_path="../data/tokenizer_math.json",
        max_length=128,
        batch_size=8
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" device: {device}")

    vocab_size = dataset.tokenizer.get_vocab_size()
    model = MathGPT(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 3



    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x) 
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            

            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}] | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f" Epoch {epoch+1} finish. average Loss: {avg_loss:.4f}\n")
    
    torch.save(model.state_dict(), "../data/math_gpt_weights.pth")
    print(" weight was save at: math_gpt_weights.pth")
    print(" finish")