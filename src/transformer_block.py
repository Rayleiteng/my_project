import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x