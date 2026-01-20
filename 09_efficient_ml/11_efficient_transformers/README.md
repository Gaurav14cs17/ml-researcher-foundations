<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%2011%20Efficient%20Transformers&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 11: Efficient Transformers

[← Back to Course](../) | [← Previous](../10_mcunet_tinyml/) | [Next: Efficient Training →](../12_efficient_training/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/11_efficient_transformers/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=2d9ZfRVaKFU&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=11) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **efficient transformer architectures**:

- **Attention complexity**: Understanding O(N²) bottleneck
- **Sparse attention**: Local, strided, and dilated patterns
- **Linear attention**: Reformulating attention to O(N)
- **FlashAttention**: Memory-efficient exact attention through tiling
- **KV Cache**: Essential optimization for autoregressive generation
- **MQA/GQA**: Reducing KV cache memory with shared keys/values

> 💡 *"FlashAttention achieves 5× speedup not by changing the math, but by optimizing memory access patterns."* — Prof. Song Han

---

![Overview](overview.png)

## The Transformer Efficiency Problem

Standard attention is O(N²) in sequence length:

| Sequence Length | Attention FLOPs | Memory |
|----------------|-----------------|--------|
| 512 | 0.26M | 1MB |
| 2048 | 4.2M | 16MB |
| 8192 | 67M | 256MB |
| 32768 | 1B | 4GB |

---

## 📐 Mathematical Foundations & Proofs

### Standard Self-Attention

**Attention mechanism:**
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

**Complexity analysis:**

For \( Q, K, V \in \mathbb{R}^{N \times d} \):
- \( QK^T \): \( O(N^2 d) \) FLOPs, \( O(N^2) \) memory
- Softmax: \( O(N^2) \) FLOPs
- Attention × V: \( O(N^2 d) \) FLOPs

**Total:** \( O(N^2 d) \) time, \( O(N^2) \) memory

---

### FlashAttention Algorithm

**Key insight:** Standard attention is memory-bound, not compute-bound.

**Problem:** Writing \( N \times N \) attention matrix to HBM is slow.

**Solution:** Compute attention in blocks, never materialize full matrix.

**Algorithm:**

```
For each query block Qi (size B_r × d):
    For each key-value block (Kj, Vj) (size B_c × d):
        Sij = Qi @ Kj.T / sqrt(d)      # In SRAM
        Pij = softmax(Sij)              # In SRAM
        Oi += Pij @ Vj                  # Accumulate
```

**Memory:** \( O(N) \) instead of \( O(N^2) \)

**I/O Complexity:**

Standard: \( O(Nd + N^2) \) HBM accesses
FlashAttention: \( O(N^2 d^2 / M) \) HBM accesses

where \( M \) is SRAM size.

---

### Safe Softmax with Online Computation

**Challenge:** Softmax requires knowing max over full row for numerical stability.

**Standard:**
```math
p_i = \frac{\exp(x_i - \max_j x_j)}{\sum_k \exp(x_k - \max_j x_j)}
```

**Online algorithm (for tiled computation):**

Maintain running \( m \) (max) and \( \ell \) (sum of exp):
```
For each new block:
    m_new = max(m_old, max(block))
    ℓ_new = exp(m_old - m_new) * ℓ_old + sum(exp(block - m_new))
    Output_new = exp(m_old - m_new) * Output_old + exp(block - m_new) @ V_block
```

**Proof of correctness:** By induction on blocks.

---

### Linear Attention

**Rewrite standard attention:**
```math
\text{Attention}(Q, K, V) = \text{softmax}(QK^T) V
```

**Linear attention approximation:**
```math
\text{Attention}(Q, K, V) \approx \phi(Q) \cdot (\phi(K)^T V)
```

where \( \phi \) is a feature map.

**Complexity:**
- Compute \( \phi(K)^T V \): \( O(Nd^2) \)
- Compute \( \phi(Q) \cdot (\phi(K)^T V) \): \( O(Nd^2) \)

**Total:** \( O(Nd^2) \) — linear in \( N \)!

**Performer kernel:**
```math
\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}} [\sin(\omega_1^T x), \cos(\omega_1^T x), ..., \sin(\omega_m^T x), \cos(\omega_m^T x)]
```

Random features approximate softmax kernel.

---

### KV Cache Analysis

**Without KV cache (inefficient):**
For each new token, recompute all K, V:
```math
K = [K_{1}, ..., K_{N}], \quad V = [V_{1}, ..., V_{N}]
```

**With KV cache:**
```math
K_{N+1} = \text{concat}(K_{cached}, k_{new})
V_{N+1} = \text{concat}(V_{cached}, v_{new})
```

**Memory requirement:**
```math
M_{KV} = 2 \times L \times N \times d \times b
```

where:
- \( L \) = number of layers
- \( N \) = sequence length
- \( d \) = head dimension
- \( b \) = bytes per element

**Example (LLaMA-7B, 2K context, FP16):**
```math
M_{KV} = 2 \times 32 \times 2048 \times 128 \times 2 = 32\text{MB per head}
```

With 32 heads: \( 32 \times 32 = 1\text{GB} \) per request!

---

### Multi-Query Attention (MQA)

**Standard MHA:** Separate K, V for each head.
```math
K_h, V_h \text{ for } h = 1, ..., H
```

**MQA:** Share K, V across all heads.
```math
K, V \text{ (single set for all heads)}
```

**Memory reduction:** \( H \times \) for KV cache.

**Accuracy trade-off:** ~1% degradation, often recoverable with fine-tuning.

---

### Grouped Query Attention (GQA)

**Middle ground:** Share KV within groups.

For \( H \) query heads and \( G \) KV groups:
```math
K_g, V_g \text{ for } g = 1, ..., G
```

Each query head \( h \) uses \( K_{\lfloor hG/H \rfloor}, V_{\lfloor hG/H \rfloor} \).

**Memory reduction:** \( H/G \times \) for KV cache.

**Example (LLaMA-2):** 32 query heads, 8 KV groups → 4× reduction.

---

### Sliding Window Attention

**Local attention pattern:**
```math
A_{ij} = \begin{cases} 
\text{softmax}(Q_i K_j^T / \sqrt{d}) & \text{if } |i - j| \leq w \\
0 & \text{otherwise}
\end{cases}
```

**Complexity:** \( O(Nw) \) instead of \( O(N^2) \).

**Effective receptive field:** After \( L \) layers, each token can attend to \( L \cdot w \) positions.

**Mistral 7B:** Uses sliding window (4K) + attention sinks.

---

### Sparse Attention Patterns

**Fixed patterns:**
- **Strided:** Attend to every \( k \)-th position
- **Local + Global:** Local window + special tokens
- **Axial:** Separate row and column attention

**Longformer pattern:**
```math
A = A_{local} + A_{global}
```

Global tokens (e.g., [CLS]) attend to all positions.

---

## 🧮 Key Derivations

### FlashAttention Speedup Analysis

**Standard attention I/O:**
```math
\text{IO}_{std} = O(Nd + N^2) = O(N^2) \text{ for } N > d
```

**FlashAttention I/O:**
```math
\text{IO}_{flash} = O\left(\frac{N^2 d^2}{M}\right)
```

**Speedup ratio:**
```math
\text{Speedup} = \frac{N^2}{N^2 d^2 / M} = \frac{M}{d^2}
```

For A100 (M = 20MB SRAM), d = 128:
```math
\text{Speedup} \approx \frac{20 \times 10^6}{128^2} \approx 1200\times \text{ I/O reduction}
```

Actual speedup: 2-4× (compute still has overhead).

---

### Memory-Compute Trade-off in KV Cache

**With KV cache:**
- Memory: \( O(NLd) \)
- Compute per token: \( O(Nd) \)

**Without KV cache:**
- Memory: \( O(Ld) \)
- Compute per token: \( O(N^2d) \)

**Break-even point:**
```math
N_{cache} \cdot Ld = N_{no-cache}^2 \cdot d
```

For large \( N \), cache always wins.

---

## 💻 Code Examples

### FlashAttention-style Tiled Computation

```python
import torch
import torch.nn.functional as F
import math

def flash_attention_forward(Q, K, V, block_size=64):
    """
    Simplified FlashAttention implementation
    Computes attention in blocks to reduce memory
    """
    batch, heads, seq_len, d = Q.shape
    
    # Initialize output and normalization
    O = torch.zeros_like(Q)
    L = torch.zeros(batch, heads, seq_len, 1, device=Q.device)  # log-sum-exp
    M = torch.full((batch, heads, seq_len, 1), float('-inf'), device=Q.device)  # max
    
    # Process in blocks
    for j in range(0, seq_len, block_size):
        j_end = min(j + block_size, seq_len)
        Kj = K[:, :, j:j_end, :]
        Vj = V[:, :, j:j_end, :]
        
        # Compute attention scores for this block
        S = torch.matmul(Q, Kj.transpose(-2, -1)) / math.sqrt(d)
        
        # Online softmax update
        M_new = torch.maximum(M, S.max(dim=-1, keepdim=True)[0])
        
        # Rescale previous output
        exp_diff = torch.exp(M - M_new)
        O = O * exp_diff
        L = L * exp_diff
        
        # Add contribution from this block
        P = torch.exp(S - M_new)
        O = O + torch.matmul(P, Vj)
        L = L + P.sum(dim=-1, keepdim=True)
        M = M_new
    
    # Normalize
    O = O / L
    return O

# KV Cache implementation
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation
    """
    def __init__(self, max_seq_len, n_layers, n_heads, head_dim, dtype=torch.float16):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        
        # Pre-allocate cache
        self.k_cache = torch.zeros(
            n_layers, 1, n_heads, max_seq_len, head_dim, dtype=dtype
        )
        self.v_cache = torch.zeros(
            n_layers, 1, n_heads, max_seq_len, head_dim, dtype=dtype
        )
        self.seq_len = 0
    
    def update(self, layer_idx, k, v):
        """Update cache with new key-value pairs"""
        new_len = k.shape[2]
        
        self.k_cache[layer_idx, :, :, self.seq_len:self.seq_len+new_len, :] = k
        self.v_cache[layer_idx, :, :, self.seq_len:self.seq_len+new_len, :] = v
        
        if layer_idx == self.n_layers - 1:
            self.seq_len += new_len
        
        return (
            self.k_cache[layer_idx, :, :, :self.seq_len, :],
            self.v_cache[layer_idx, :, :, :self.seq_len, :]
        )
    
    def clear(self):
        self.seq_len = 0

# Multi-Query Attention
class MultiQueryAttention(torch.nn.Module):
    """
    Multi-Query Attention: Single KV head, multiple query heads
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, self.head_dim)  # Single head
        self.v_proj = torch.nn.Linear(d_model, self.head_dim)  # Single head
        self.o_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x, kv_cache=None):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, 1, self.head_dim)
        
        # Expand K, V to match Q heads
        k = k.expand(-1, -1, self.n_heads, -1)
        v = v.expand(-1, -1, self.n_heads, -1)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)

# Sliding Window Attention
def sliding_window_attention(Q, K, V, window_size=256):
    """
    Attention with sliding window - O(n*w) complexity
    """
    batch, heads, seq_len, d = Q.shape
    
    # Create windowed attention mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False
    
    # Standard attention with mask
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    
    return torch.matmul(attn, V)
```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| FlashAttention | All modern LLM training |
| MQA/GQA | LLaMA 2, Mistral inference |
| Sliding Window | Long document processing |
| PagedAttention | vLLM serving |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← MCUNet & TinyML](../10_mcunet_tinyml/README.md) | [Efficient ML](../README.md) | [Efficient Training →](../12_efficient_training/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | FlashAttention | [arXiv](https://arxiv.org/abs/2205.14135) |
| 📄 | FlashAttention-2 | [arXiv](https://arxiv.org/abs/2307.08691) |
| 📄 | Longformer | [arXiv](https://arxiv.org/abs/2004.05150) |
| 📄 | Multi-Query Attention | [arXiv](https://arxiv.org/abs/1911.02150) |
| 📄 | Performer | [arXiv](https://arxiv.org/abs/2009.14794) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
