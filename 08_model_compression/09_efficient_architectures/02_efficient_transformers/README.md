<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00D9FF&height=100&section=header&text=Efficient%20Transformers&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08.09.02-00D9FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### 1. Standard Attention Complexity

**Self-Attention:**

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Where $Q, K, V \in \mathbb{R}^{n \times d}$

**Complexity Analysis:**
- $QK^T$: $O(n^2 d)$ operations, $O(n^2)$ memory
- Softmax: $O(n^2)$
- $(\cdot)V$: $O(n^2 d)$

**Total:** $O(n^2 d)$ compute, $O(n^2)$ memory

**Problem:** For $n = 8192$: 67M attention scores per head!

### 2. Linformer: Low-Rank Projection

**Key Insight:** Attention matrix is approximately low-rank.

**Projection:**

```math
\tilde{K} = E_K K, \quad \tilde{V} = E_V V
```

Where $E\_K, E\_V \in \mathbb{R}^{k \times n}$ project to $k \ll n$ dimensions.

**New Attention:**

```math
\text{Attention} = \text{softmax}\left(\frac{Q\tilde{K}^T}{\sqrt{d}}\right)\tilde{V}
```

**Complexity:** $O(nkd)$ instead of $O(n^2d)$

**Theorem (Johnson-Lindenstrauss):**
For any $\epsilon > 0$, random projection preserves distances with high probability:

```math
\|E_K x - E_K y\|_2 \approx (1 \pm \epsilon)\|x - y\|_2
```

### 3. Performer: Kernel Approximation

**Standard Attention:**

```math
\text{Att}(q, K, V) = \frac{\sum_i \exp(q^T k_i) v_i}{\sum_i \exp(q^T k_i)}
```

**Kernel View:**

```math
\text{Att}(q, K, V) = \frac{\sum_i \kappa(q, k_i) v_i}{\sum_i \kappa(q, k_i)}
```

Where $\kappa(q, k) = \exp(q^T k)$ (softmax kernel)

**FAVOR+ Approximation:**

```math
\kappa(q, k) \approx \phi(q)^T \phi(k)
```

Using random features:

```math
\phi(x) = \frac{1}{\sqrt{m}}\left[\exp\left(w_1^T x - \frac{\|x\|^2}{2}\right), ..., \exp\left(w_m^T x - \frac{\|x\|^2}{2}\right)\right]
```

**Linear Attention:**

```math
\text{Att}(Q, K, V) = \phi(Q)\left(\phi(K)^T V\right)
```

Compute $\phi(K)^T V$ first: $O(md)$ per position → $O(nmd)$ total

### 4. Flash Attention: IO-Aware Algorithm

**Problem:** Standard attention is memory-bound.

**Memory Hierarchy:**
- HBM (GPU memory): Large (40GB), slow (2TB/s)
- SRAM (on-chip): Small (20MB), fast (19TB/s)

**Standard Attention Memory Access:**
1. Load $Q, K$ from HBM → Compute $QK^T$ → Store to HBM: $O(n^2)$
2. Load $QK^T$ → Softmax → Store: $O(n^2)$
3. Load softmax, $V$ → Output: $O(n^2)$

**Flash Attention Algorithm:**
- Tile $Q, K, V$ into blocks that fit in SRAM
- Compute attention for each block pair
- Use online softmax (track running max and sum)
- Never materialize full $n \times n$ attention matrix

**IO Complexity:**

```math
\text{Standard: } O(n^2 d + n^2)
\text{Flash: } O(n^2 d^2 / M)
```

Where $M$ = SRAM size. Typical reduction: $10-100\times$

### 5. Grouped Query Attention (GQA)

**Multi-Head Attention:**

```math
\text{MHA}(X) = \text{Concat}(head_1, ..., head_h)W^O
head_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)
```

**Memory:** $h$ sets of K,V projections

**Multi-Query Attention (MQA):**
Share K,V across all heads:

```math
head_i = \text{Attention}(XW_Q^i, XW_K, XW_V)
```

**Memory:** 1 set of K,V projections (but quality degrades)

**Grouped Query Attention (GQA):**
Share K,V within groups of heads:

```math
head_i = \text{Attention}(XW_Q^i, XW_K^{g(i)}, XW_V^{g(i)})
```

Where $g(i)$ maps head $i$ to its group.

**KV Cache Reduction:**

```math
\text{MHA: } O(h \cdot d_k \cdot L)
\text{GQA: } O(G \cdot d_k \cdot L)
\text{MQA: } O(d_k \cdot L)
```

### 6. Sliding Window Attention

**Local Attention:**

```math
A_{ij} = \begin{cases} \text{softmax}(\cdot) & |i-j| \leq w \\ 0 & \text{otherwise} \end{cases}
```

**Complexity:** $O(nw)$ instead of $O(n^2)$

**Effective Receptive Field:**
After $L$ layers: token can attend to $L \cdot w$ positions (linear growth)

**Longformer:** Local + Global attention
- Most tokens: Local window $w$
- Special tokens (CLS): Global attention

---

## 🎯 The Problem and Solutions

```
Standard Attention: O(n²)

Attention(Q,K,V) = softmax(QK^T / √d) V
                   -----------------
                   n×n matrix!

For n = 8192: 67 million scores per head!
For n = 100K: 10 billion scores! (impossible)

Solutions:
+-------------+------------+--------------+
|   Method    | Complexity |   Key Idea   |
+-------------+------------+--------------+
| Linformer   |   O(nk)    | Project K,V  |
| Performer   |   O(nm)    | Kernel trick |
| Flash Attn  |   O(n²)    | IO-aware     |
| Local Attn  |   O(nw)    | Window only  |
| GQA/MQA     |   O(n²)    | Share KV     |
+-------------+------------+--------------+
```

---

## 💻 Code Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== Linear Attention (Performer-style) ==========
class LinearAttention(nn.Module):
    """Linear attention using kernel feature map"""
    
    def __init__(self, d_model, n_heads, feature_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.feature_dim = feature_dim
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Random features for kernel approximation
        self.random_features = nn.Parameter(
            torch.randn(self.d_k, feature_dim), requires_grad=False
        )
    
    def feature_map(self, x):
        """φ(x) using random Fourier features"""
        # x: [B, H, L, d_k]
        projection = x @ self.random_features  # [B, H, L, feature_dim]
        return torch.exp(projection - x.pow(2).sum(-1, keepdim=True) / 2)
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply feature map
        Q_prime = self.feature_map(Q)  # [B, H, L, m]
        K_prime = self.feature_map(K)  # [B, H, L, m]
        
        # Linear attention: φ(Q) @ (φ(K)^T @ V)
        KV = torch.einsum('bhld,bhlv->bhdv', K_prime, V)  # [B, H, m, d_k]
        out = torch.einsum('bhld,bhdv->bhlv', Q_prime, KV)  # [B, H, L, d_k]
        
        # Normalize
        K_sum = K_prime.sum(dim=2, keepdim=True)  # [B, H, 1, m]
        normalizer = torch.einsum('bhld,bhmd->bhl', Q_prime, K_sum)  # [B, H, L]
        out = out / (normalizer.unsqueeze(-1) + 1e-6)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_o(out)

# ========== Grouped Query Attention ==========
class GroupedQueryAttention(nn.Module):
    """GQA: Share K,V across head groups"""
    
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K,V for each head group
        K = K.repeat_interleave(self.n_rep, dim=1)  # [B, n_heads, L, d_k]
        V = V.repeat_interleave(self.n_rep, dim=1)
        
        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_o(out)

# ========== Sliding Window Attention ==========
class SlidingWindowAttention(nn.Module):
    """Local attention with fixed window size"""
    
    def __init__(self, d_model, n_heads, window_size=256):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, L, D = x.shape
        w = self.window_size
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create sliding window mask
        mask = torch.ones(L, L, device=x.device)
        for i in range(L):
            start = max(0, i - w // 2)
            end = min(L, i + w // 2 + 1)
            mask[i, :start] = 0
            mask[i, end:] = 0
        
        # Attention with mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.W_o(out)
```

---

## 📊 Comparison

| Method | Complexity | Quality | Long Context |
|--------|------------|---------|--------------|
| Standard | O(n²) | Best | ≤8K tokens |
| Linformer | O(nk) | Good | 100K+ tokens |
| Performer | O(nm) | Good | Unlimited |
| Flash Attention | O(n²) | Best | ~128K tokens |
| Local + Global | O(nw) | Good | Long docs |
| GQA | O(n²) | Good | Inference speedup |

---

## 📐 Complexity Analysis Visualization

<img src="./images/attention-complexity.svg" width="100%">

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Linformer](https://arxiv.org/abs/2006.04768) | Wang et al. | 2020 | O(n) via projection |
| [Performer](https://arxiv.org/abs/2009.14794) | Choromanski et al. | 2020 | FAVOR+ random features |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | Dao et al. | 2022 | IO-aware attention |
| [Flash Attention 2](https://arxiv.org/abs/2307.08691) | Dao | 2023 | 2× faster |
| [Longformer](https://arxiv.org/abs/2004.05150) | Beltagy et al. | 2020 | Local + global |
| [GQA](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 2023 | Grouped queries |

---

⬅️ [Back: Efficient Networks](../01_efficient_networks/README.md) | ➡️ [Next: Compression Pipelines](../../10_compression_pipelines/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
