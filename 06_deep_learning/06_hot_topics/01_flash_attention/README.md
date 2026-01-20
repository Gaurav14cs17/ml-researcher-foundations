<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Flash%20Attention&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 The Problem

Standard attention has a memory bottleneck:

```math
\text{Given: } Q, K, V \in \mathbb{R}^{n \times d}
\text{Compute: } O = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V

```

### Standard Algorithm:

```
1. S = QK^T           → n × n matrix (materialize!)

2. A = softmax(S)     → n × n matrix (materialize!)  

3. O = AV             → n × d matrix

Memory: O(n²) - prohibitive for long sequences!

```

**Real-world impact:**
- GPT-2/3: Limited to 2K-4K tokens

- Cannot process long documents, codebases, conversations

- Memory becomes bottleneck before computation

---

## 📐 Mathematical Foundation

### GPU Memory Hierarchy

```
+-------------------------------------+
|  SRAM (on-chip)                     |
|  • Fast: ~19 TB/s bandwidth         |
|  • Small: ~20 MB                    |
+-------------------------------------+
         ↕ (expensive data movement!)
+-------------------------------------+

|  HBM (High Bandwidth Memory)        |
|  • Slower: ~1.5 TB/s bandwidth      |
|  • Large: ~40 GB                    |
+-------------------------------------+

```

**Standard attention:** Constantly moves $n^2$ matrix between HBM ↔ SRAM  
**Flash Attention:** Never materializes full $n^2$ matrix, stays in SRAM

---

## 🔬 The Key Insight: Online Softmax

### Problem: Softmax Needs All Values

Standard softmax:

```math
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

```

Needs global sum → can't compute in blocks!

### Solution: Online Softmax Algorithm

**Key observation:** Softmax can be computed iteratively.

For vectors $x^{(1)}$ and $x^{(2)}$:

```math
m^{(1)} = \max(x^{(1)}), \quad m^{(2)} = \max(x^{(2)})
m^{\text{new}} = \max(m^{(1)}, m^{(2)})
\ell^{(1)} = \sum_i e^{x^{(1)}_i - m^{(1)}}, \quad \ell^{(2)} = \sum_i e^{x^{(2)}_i - m^{(2)}}
\ell^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \ell^{(1)} + e^{m^{(2)} - m^{\text{new}}} \ell^{(2)}

```

This allows processing in blocks while maintaining numerical stability!

---

## 📐 Flash Attention Algorithm

### Tiling Strategy

```python
# Conceptual overview
Block_size = B = SRAM_size / (3 * d)  # Fit Q, K, V blocks

# Divide Q, K, V into blocks
Q_blocks = split(Q, B)  # T_q blocks
K_blocks = split(K, B)  # T_k blocks  
V_blocks = split(V, B)

# Initialize
O = zeros(n, d)
L = zeros(n)      # Row sum of exp
M = -inf * ones(n) # Row max

for i in range(T_q):
    load Q_i into SRAM
    
    for j in range(T_k):
        load K_j, V_j into SRAM
        
        # Compute local attention
        S_ij = Q_i @ K_j.T / sqrt(d)  # B × B block
        
        # Online softmax update
        m_new = max(M[i*B:(i+1)*B], rowmax(S_ij))
        exp_S = exp(S_ij - m_new)
        l_new = exp(M - m_new) * L + rowsum(exp_S)
        
        # Update output
        O_i = diag(exp(M - m_new) / l_new) @ O_i + exp_S @ V_j / l_new
        
        M = m_new
        L = l_new
    
    write O_i to HBM

# Result: Exact attention, O(n) memory!

```

### Memory Complexity Analysis

| Method | Memory | IO Complexity |
|--------|--------|---------------|
| Standard | $O(n^2)$ | $O(n^2 + nd)$ reads/writes |
| Flash Attention | $O(n)$ | $O(n^2 d^2 / M)$ reads |

Where $M$ = SRAM size.

**Key insight:** Trades extra FLOPs for fewer memory accesses.

---

## 🔬 Backward Pass

### Challenge: Standard Backprop Needs Stored Attention Matrix

Forward: Store $A = \text{softmax}(QK^\top/\sqrt{d})$ for backward.  
This is $O(n^2)$ memory!

### Flash Attention Solution: Recomputation

Instead of storing $A$:

1. Store only $O$, $L$ (row sums), $M$ (row maxes)

2. Recompute $A$ during backward pass in tiled fashion

**Memory:** $O(n)$ instead of $O(n^2)$  
**Compute:** ~20% extra FLOPs, but net win due to reduced IO

### Gradient Formulas

Given $\frac{\partial L}{\partial O}$:

```math
\frac{\partial L}{\partial V} = A^\top \frac{\partial L}{\partial O}
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^\top

```

For softmax gradient (let $P = \frac{\partial L}{\partial A}$):

```math
\frac{\partial L}{\partial S} = A \odot (P - \text{rowsum}(A \odot P))
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d}} \frac{\partial L}{\partial S} K
\frac{\partial L}{\partial K} = \frac{1}{\sqrt{d}} \left(\frac{\partial L}{\partial S}\right)^\top Q

```

---

## 💻 Implementation

### PyTorch 2.0+

```python
import torch
import torch.nn.functional as F

# Shapes: (batch, n_heads, seq_len, head_dim)
q = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)

# Automatically uses Flash Attention if available!
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True  # For autoregressive models
)

print(f"Output shape: {output.shape}")

```

### Flash Attention Library

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

# Shape: (batch, seqlen, nheads, headdim)
q = torch.randn(2, 4096, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 8, 64, device='cuda', dtype=torch.float16)

output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,
    window_size=(-1, -1),  # Full attention
    return_attn_probs=False
)

# For variable-length sequences
from flash_attn import flash_attn_varlen_func

# Packed sequences with cumulative lengths
output = flash_attn_varlen_func(
    q.view(-1, 8, 64),  # Flatten batch and seq
    k.view(-1, 8, 64),
    v.view(-1, 8, 64),
    cu_seqlens_q=torch.tensor([0, 4096, 8192], dtype=torch.int32, device='cuda'),
    cu_seqlens_k=torch.tensor([0, 4096, 8192], dtype=torch.int32, device='cuda'),
    max_seqlen_q=4096,
    max_seqlen_k=4096,
    causal=True
)

```

### Memory Comparison

```python
import torch
import gc

def measure_memory(seq_len, use_flash=False):
    torch.cuda.empty_cache()
    gc.collect()
    
    q = torch.randn(1, 8, seq_len, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, seq_len, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, seq_len, 64, device='cuda', dtype=torch.float16)
    
    torch.cuda.reset_peak_memory_stats()
    
    if use_flash:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        # Standard attention
        scores = (q @ k.transpose(-2, -1)) / 8.0
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v
    
    return torch.cuda.max_memory_allocated() / 1e9  # GB

# Compare
for seq_len in [1024, 4096, 8192, 16384]:
    try:
        standard = measure_memory(seq_len, use_flash=False)
    except:
        standard = float('inf')
    
    flash = measure_memory(seq_len, use_flash=True)
    print(f"Seq {seq_len}: Standard={standard:.2f}GB, Flash={flash:.2f}GB")

```

---

## 📊 Performance Benchmarks

### Memory Complexity

| Method | Memory | Can Handle |
|--------|--------|------------|
| Standard | $O(n^2)$ | ~2K tokens |
| Flash Attention | $O(n)$ | 100K+ tokens |
| Flash Attention 2 | $O(n)$ | 100K+ tokens, 2x faster |

### Speed Benchmarks (A100 GPU)

| Sequence Length | Standard | Flash 1 | Flash 2 |
|-----------------|----------|---------|---------|
| 512 | 1.0x | 2.5x | 3.5x |
| 2K | 1.0x | 3.0x | 5.0x |
| 8K | OOM | 3.5x | 6.5x |
| 32K | OOM | 4.0x | 7.0x |

---

## 🌍 Real-World Impact

### Models Using Flash Attention

| Model | Context Length | Notes |
|-------|----------------|-------|
| **GPT-4** | 8K-128K | Long context via Flash |
| **Claude 3** | 200K | Extreme long context |
| **LLaMA 2** | 4K | Extended to 32K with Flash |
| **Mistral** | 8K-32K | Native sliding window |
| **MPT** | 65K | One of first to advertise |
| **Code models** | 100K+ | Full file contexts |

### Applications Enabled

1. **Long Document QA:** Process full papers, books

2. **Code Understanding:** Analyze entire repositories  

3. **Long-form Generation:** Write coherent long articles

4. **Conversation History:** Remember full chat context

---

## 🔄 Variants and Extensions

### Flash Attention 2 (2023)

- 2x faster than Flash 1
- Better parallelization across sequence length

- Tuned for A100/H100 architecture

### Flash Attention 3 (2024)

- Support for:
  - Multi-query attention (MQA)
  - Grouped-query attention (GQA)  
  - Sparse attention patterns
  - Block-diagonal masks

### PagedAttention (vLLM)

- Complementary technique

- Optimizes KV cache management for inference

- Combined with Flash for efficient serving

### Ring Attention

- Extends Flash Attention across multiple GPUs

- Enables million-token contexts

- Overlaps communication with computation

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Flash Attention Paper | [arXiv](https://arxiv.org/abs/2205.14135) |
| 📄 | Flash Attention 2 | [arXiv](https://arxiv.org/abs/2307.08691) |
| 📄 | Online Softmax | [Paper](https://arxiv.org/abs/1805.02867) |
| 💻 | GitHub | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 📖 | PyTorch SDPA | [Docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| 🎥 | Horace He: Making DL Go Brrrr | [Blog](https://horace.io/brrr_intro.html) |
| 🇨🇳 | Flash Attention详解 | [知乎](https://zhuanlan.zhihu.com/p/638468472) |
| 🇨🇳 | 注意力加速原理 | [CSDN](https://blog.csdn.net/v_JULY_v/article/details/133619540) |

---

➡️ [Next: LoRA](../02_lora/README.md)

---

⬅️ [Back: Hot Topics](../../README.md)

---

➡️ [Next: LoRA](../02_lora/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
