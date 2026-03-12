import torch
import torch.nn as nn
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("../data/tokenizer_math.json")
vocab_size = tokenizer.get_vocab_size()

class MathTextbookEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=256, max_seq_length=1024):
        super().__init__()
        
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_length, embedding_dim=d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)
        
        token_embs = self.token_embedding(input_ids)
        pos_embs = self.position_embedding(positions)
        
        embeddings = token_embs + pos_embs
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

if __name__ == "__main__":
    d_model = 256
    embedding_layer = MathTextbookEmbedding(vocab_size=vocab_size, d_model=d_model)
    
    test_formula = r"\int_{-\infty}^{\infty} \hat{f}(\xi) d\xi"
    
    encoded = tokenizer.encode(test_formula)
    token_ids = encoded.ids
    print(f"\n Input Token IDs: {token_ids}")
    
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    print(f" PyTorch Tensor shape: {input_tensor.shape}")
    
    out_vectors = embedding_layer(input_tensor)
    
    print(f"\n Final Embedding output shape: {out_vectors.shape}")
    print(f"explain: (Batch Size={out_vectors.shape[0]}, Sequence Length={out_vectors.shape[1]}, Hidden Dimension={out_vectors.shape[2]})")