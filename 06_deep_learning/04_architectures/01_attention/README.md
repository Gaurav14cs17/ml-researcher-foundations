<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Attention%20Mechanisms&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:

- $Q \in \mathbb{R}^{n \times d_k}$ (queries)

- $K \in \mathbb{R}^{m \times d_k}$ (keys)

- $V \in \mathbb{R}^{m \times d_v}$ (values)

- Output: $\mathbb{R}^{n \times d_v}$

### Why Scale by $\sqrt{d_k}$?

**Variance Analysis:**

If $q_i, k_j \sim \mathcal{N}(0, 1)$ independently:

$$\text{Var}(q^\top k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

Large variance → softmax becomes very peaked → vanishing gradients.

Scaling by $\sqrt{d_k}$: $\text{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) = 1$

---

## 📐 Attention Score Interpretation

### As Weighted Average

$$\text{output}_i = \sum_j \alpha_{ij} v_j$$

Where $\alpha_{ij} = \text{softmax}\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)_j$

**Properties:**
- $\sum_j \alpha_{ij} = 1$ (weights sum to 1)

- $\alpha_{ij} \geq 0$ (non-negative weights)

- Differentiable weighted average

### As Soft Retrieval

$$\text{score}(q, k) = q^\top k$$

High score = high relevance = more weight in output.

---

## 📐 Types of Attention

### 1. Scaled Dot-Product (Transformer)

$$\text{score}(q, k) = \frac{q^\top k}{\sqrt{d_k}}$$

**Complexity:** $O(n \cdot m \cdot d_k)$

### 2. Additive/Bahdanau Attention

$$\text{score}(q, k) = v^\top \tanh(W_q q + W_k k)$$

Where $W_q, W_k, v$ are learnable parameters.

**Advantages:** More flexible for different-sized $q, k$.

### 3. Multiplicative/Luong Attention

$$\text{score}(q, k) = q^\top W k$$

Where $W$ is learnable.

### 4. Relative Position Attention

$$\text{score}(q_i, k_j) = q_i^\top k_j + q_i^\top r_{i-j}$$

Where $r_{i-j}$ is a learnable relative position embedding.

---

## 📐 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Projections

- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$

- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$

- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$

- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### Parameters

$$\text{Total params} = h(d_{model} \cdot d_k + d_{model} \cdot d_k + d_{model} \cdot d_v) + hd_v \cdot d_{model}$$

With $d_k = d_v = d_{model}/h$:

$$= 4 \cdot d_{model}^2$$

### Why Multiple Heads?

Different heads can attend to different aspects:

- Position information

- Syntactic relationships

- Semantic similarity

- Copy patterns

---

## 📐 Self-Attention vs Cross-Attention

### Self-Attention

$Q, K, V$ all come from the same sequence $X$:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

**Use:** Transformers encoder, GPT decoder

### Cross-Attention

$Q$ from one sequence, $K, V$ from another:

$$Q = XW^Q, \quad K = YW^K, \quad V = YW^V$$

**Use:** Encoder-decoder (translation), CLIP, Stable Diffusion

---

## 📐 Masked Attention

### Causal Masking (Autoregressive)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

Where $M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$

**Effect:** Position $i$ can only attend to positions $\leq i$.

### Padding Masking

Mask out padding tokens to prevent attention to them.

---

## 📐 Attention Gradients

### Forward

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
O = AV$$

### Backward

Given $\frac{\partial \mathcal{L}}{\partial O}$:

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}
\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} V^\top$$

For softmax with scores $S = QK^\top / \sqrt{d_k}$:

$$\frac{\partial \mathcal{L}}{\partial S_{ij}} = A_{ij}\left(\frac{\partial \mathcal{L}}{\partial A_{ij}} - \sum_k A_{ik}\frac{\partial \mathcal{L}}{\partial A_{ik}}\right)$$

Then:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S} K
\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S}^\top Q$$

---

## 📊 Complexity Analysis

| Operation | Time | Memory |
|-----------|------|--------|
| $QK^\top$ | $O(n^2 d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| $\text{Attention} \cdot V$ | $O(n^2 d)$ | $O(nd)$ |
| **Total** | **$O(n^2 d)$** | **$O(n^2)$** |

For sequence length $n = 4096$, $d = 1024$:

- Attention matrix: $4096^2 \times 4 \text{ bytes} = 64\text{ MB}$ per layer per head!

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, n_heads, seq_q, d_k)
        K: (batch, n_heads, seq_k, d_k)
        V: (batch, n_heads, seq_k, d_v)
        mask: (batch, 1, seq_q, seq_k) or (1, 1, seq_q, seq_k)
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.W_o(output)
        
        return output, attn_weights

def causal_mask(seq_len, device):
    """Create causal (autoregressive) mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

# Example usage
d_model = 512
n_heads = 8
seq_len = 100
batch_size = 32

mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(batch_size, seq_len, d_model)
mask = causal_mask(seq_len, x.device)

output, attn_weights = mha(x, x, x, mask)  # Self-attention
print(f"Output shape: {output.shape}")  # (32, 100, 512)
print(f"Attention weights: {attn_weights.shape}")  # (32, 8, 100, 100)

```

---

## 🔗 Applications

| Type | Used In |
|------|---------|
| **Self-Attention** | Transformers, GPT, BERT |
| **Cross-Attention** | Encoder-Decoder, CLIP, Stable Diffusion |
| **Flash Attention** | Efficient LLMs |
| **Linear Attention** | Performer, Linear Transformer |
| **Sparse Attention** | Longformer, BigBird |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
| 📄 | Flash Attention | [arXiv](https://arxiv.org/abs/2205.14135) |
| 📄 | Bahdanau Attention | [arXiv](https://arxiv.org/abs/1409.0473) |
| 🎥 | Illustrated Transformer | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) |
| 🇨🇳 | 注意力机制详解 | [知乎](https://zhuanlan.zhihu.com/p/47063917) |

---

➡️ [Next: CNN](../02_cnn/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

➡️ [Next: CNN](../02_cnn/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
