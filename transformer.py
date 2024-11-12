import torch
import torch.nn as nn
import torch.nn.functional as F
import math


d_model = 512
num_heads = 8
drop_prob = 0.1
num_classes = 1  
ffn_hidden = 2048
num_layers = 5
max_sequence_length = 200


def scaled_dot_product(q, k, v):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        
        qkv = self.qkv_layer(x)  # Shape: [batch_size, seq_len, 3 * d_model]
        
        # Split qkv into q, k, v and reshape for multi-head attention
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Transpose for multi-head attention computation
        q = q.transpose(1, 2)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v)
        
        # Concatenate heads and apply final linear layer
        values = values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.linear_layer(values)

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# Encoder Stack
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Full Transformer Model with Encoder for Classification
class TransformerClassifier(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, num_classes, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer for token IDs
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token IDs to embeddings
        enc_out = self.encoder(x)
        enc_out = enc_out[:, 0, :]  # Take [CLS] token output for classification
        return self.classifier(enc_out)
