<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Transformer%20Architecture&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

The Transformer architecture, introduced in "Attention Is All You Need" (2017), revolutionized NLP and became the foundation for modern LLMs. It replaces recurrence with self-attention, enabling parallel computation and better long-range dependency modeling.

---

## 📐 Self-Attention Mechanism

### Scaled Dot-Product Attention

**Input:** Query $Q$, Key $K$, Value $V$ matrices

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

**Dimensions:**
- $Q \in \mathbb{R}^{n \times d_k}$
- $K \in \mathbb{R}^{m \times d_k}$
- $V \in \mathbb{R}^{m \times d_v}$
- Output: $\mathbb{R}^{n \times d_v}$

### Why Scaling by $\sqrt{d_k}$?

**Problem:** For large $d_k$, dot products grow large, pushing softmax into saturation.

**Proof:**
```
Assume q, k ~ N(0, 1) independently

E[q·k] = E[Σᵢ qᵢkᵢ] = Σᵢ E[qᵢ]E[kᵢ] = 0

Var[q·k] = Var[Σᵢ qᵢkᵢ] = Σᵢ Var[qᵢkᵢ]
         = Σᵢ E[qᵢ²]E[kᵢ²] = dₖ · 1 · 1 = dₖ

So q·k ~ N(0, dₖ) approximately

For large dₖ, values can be very large (e.g., ±10 for dₖ=64)
softmax(10) ≈ 1, gradients vanish!

Solution: Divide by √dₖ to get q·k/√dₖ ~ N(0, 1)
```

### Attention as Soft Dictionary Lookup

```
Interpretation:
- Keys K: What information is available at each position
- Queries Q: What information does each position want
- Values V: Actual content to retrieve

q·kᵢ measures "relevance" of position i to query q
softmax normalizes to get "attention weights"
Output is weighted combination of values
```

---

## 🔬 Multi-Head Attention

### Formulation

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
```

where each head:

```math
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
```

**Projections:**
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

Typically: $d_k = d_v = d_{model} / h$

### Why Multiple Heads?

```
Single attention: One "query pattern" per position
Multi-head: Multiple parallel attention patterns

Example with h=8 heads:
- Head 1: Attends to previous word
- Head 2: Attends to subject
- Head 3: Attends to verb
- Head 4: Attends to syntactic structure
...

Each head learns different relationship types!
```

---

## 📊 Positional Encoding

### Sinusoidal Encoding

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

### Why Sinusoidal?

**Property 1: Unique encoding for each position**
```
Different positions have different PE vectors
PE(pos) ≠ PE(pos') for pos ≠ pos'
```

**Property 2: Relative positions via linear transformation**
```
For any offset k, there exists a matrix M such that:
PE(pos + k) = M · PE(pos)

Proof:
PE uses sin(ωᵢ·pos) and cos(ωᵢ·pos)
Using rotation identity:
sin(ωᵢ(pos+k)) = sin(ωᵢpos)cos(ωᵢk) + cos(ωᵢpos)sin(ωᵢk)
cos(ωᵢ(pos+k)) = cos(ωᵢpos)cos(ωᵢk) - sin(ωᵢpos)sin(ωᵢk)

This is a linear transformation (2×2 rotation per frequency)!
Model can learn to attend to relative positions.
```

### Rotary Position Embedding (RoPE)

Modern alternative used in LLaMA, GPT-NeoX:

```math
f_q(x_m, m) = (W_q x_m) e^{im\theta}
f_k(x_n, n) = (W_k x_n) e^{in\theta}
```

**Key property:** Attention only depends on relative position $m - n$:

```math
\langle f_q(x_m, m), f_k(x_n, n) \rangle = \text{Re}[(W_q x_m)(W_k x_n)^* e^{i(m-n)\theta}]
```

---

## 📐 Transformer Block

### Architecture

```
Input x
    |
    +----------------------+
    ▼                      |
LayerNorm                  |
    |                      |
Multi-Head Attention       |
    |                      |
    ▼                      |
    + ◄--------------------+  (residual)
    |
    +----------------------+
    ▼                      |
LayerNorm                  |
    |                      |
Feed-Forward Network       |
    |                      |
    ▼                      |
    + ◄--------------------+  (residual)
    |
Output
```

### Pre-LN vs Post-LN

**Post-LN (Original):**

```math
x = x + \text{Attn}(x)
x = \text{LN}(x)
```

**Pre-LN (Modern):**

```math
x = x + \text{Attn}(\text{LN}(x))
```

Pre-LN is more stable for deep networks (gradient scale is bounded).

### Feed-Forward Network

```math
\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x + b_1) + b_2
```

**Dimensions:**
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- Typical: $d_{ff} = 4 \cdot d_{model}$

**Activation evolution:**
- Original: ReLU
- BERT: GELU
- LLaMA: SwiGLU

---

## 🎯 Attention Variants

### Causal (Masked) Attention

For autoregressive generation, prevent attending to future:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
```

where mask $M_{ij} = -\infty$ if $j > i$, else 0.

### Cross-Attention

Encoder-decoder models:
- Q from decoder
- K, V from encoder

```math
\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc})
```

### Efficient Attention Variants

| Method | Complexity | Description |
|--------|------------|-------------|
| **Standard** | $O(n^2d)$ | Full attention matrix |
| **Sparse** | $O(n\sqrt{n}d)$ | Fixed sparse patterns |
| **Linear** | $O(nd^2)$ | Kernel approximation |
| **Flash Attention** | $O(n^2d)$ | Memory-efficient |

---

## 💻 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, n_heads, seq_len, d_k)
            k: (batch, n_heads, seq_len, d_k)
            v: (batch, n_heads, seq_len, d_v)
            mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        
        Returns:
            output: (batch, n_heads, seq_len, d_v)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        d_k = q.size(-1)
        
        # Compute attention scores: QK^T / √d_k
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (for causal attention or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over last dimension (keys)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: (batch, seq_len, d_model)
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = q.size(0)
        
        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    """
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # Precompute rotation angles
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cos/sin cache
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = W_2 · activation(W_1 x + b_1) + b_2
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()  # SwiGLU uses SiLU
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class SwiGLU(nn.Module):
    """
    SwiGLU activation (used in LLaMA)
    SwiGLU(x) = (xW₁) ⊙ Swish(xV)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """
    Single Transformer Block (Pre-LN variant)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        # Self-attention with residual
        attn_output, _ = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x

class Transformer(nn.Module):
    """
    Complete Transformer (Encoder-only, like BERT/GPT)
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) token indices
            mask: attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """

        # Embed tokens and add positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.ln_final(x)
    
    @staticmethod
    def create_causal_mask(seq_len, device):
        """Create causal (autoregressive) mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True where attention is allowed

class GPTModel(nn.Module):
    """
    GPT-style Language Model
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12,
                 d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        
        self.transformer = Transformer(vocab_size, d_model, n_heads, n_layers,
                                       d_ff, max_len, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.transformer.token_embedding.weight
    
    def forward(self, x):

        # Create causal mask
        seq_len = x.size(1)
        mask = Transformer.create_causal_mask(seq_len, x.device)
        
        # Transformer forward
        hidden = self.transformer(x, mask)
        
        # Language model head
        logits = self.lm_head(hidden)
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
        """
        Autoregressive generation
        """
        for _ in range(max_new_tokens):

            # Get logits for last position
            logits = self(prompt)[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            prompt = torch.cat([prompt, next_token], dim=1)
        
        return prompt

# Example usage
vocab_size = 50000
model = GPTModel(vocab_size, d_model=256, n_heads=8, n_layers=4, d_ff=1024)

# Forward pass
input_ids = torch.randint(0, vocab_size, (2, 128))
logits = model(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")  # (2, 128, 50000)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.2f}M")
```

---

## 📊 Complexity Analysis

| Component | Time | Space |
|-----------|------|-------|
| Self-Attention | $O(n^2 d)$ | $O(n^2 + nd)$ |
| FFN | $O(n d d_{ff})$ | $O(n d_{ff})$ |
| Full Layer | $O(n^2 d + n d d_{ff})$ | $O(n^2 + n d_{ff})$ |

---

## 📚 Key Insights

| Insight | Details |
|---------|---------|
| **Scaling** | √d scaling prevents gradient issues |
| **Multi-head** | Parallel attention heads capture different patterns |
| **Residuals** | Enable training of very deep models |
| **Pre-LN** | More stable than post-LN for deep models |
| **RoPE > Sinusoidal** | Better extrapolation to longer sequences |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Attention Is All You Need | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) |
| 📄 | RoFormer (RoPE) | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| 📄 | Flash Attention | [Dao et al., 2022](https://arxiv.org/abs/2205.14135) |
| 🎥 | Illustrated Transformer | [jalammar.github.io](http://jalammar.github.io/illustrated-transformer/) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Seq2Seq](../10_seq2seq/README.md) | [Architectures](../README.md) | [ViT](../12_vit/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
