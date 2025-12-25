<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Flash%20Attention&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 The Problem

Standard attention has a memory bottleneck:

```
Given: Q, K, V each n × d
Compute: output = softmax(QKᵀ)V

Steps:
1. S = QKᵀ           → n × n matrix (materialize!)
2. A = softmax(S)    → n × n matrix (materialize!)
3. O = AV            → n × d matrix

Memory: O(n²) - prohibitive for long sequences!
```

**Real-world impact:**
- GPT-2/3: Limited to 2K-4K tokens
- Cannot process long documents, codebases, conversations
- Memory becomes bottleneck before computation

---

## 🔑 Key Insight

### GPU Memory Hierarchy

```
┌─────────────────────────────────────┐
│  SRAM (on-chip)                     │
│  • Fast: ~19 TB/s bandwidth         │
│  • Small: ~20 MB                    │
└─────────────────────────────────────┘
         ↕ (expensive data movement!)
┌─────────────────────────────────────┐
│  HBM (High Bandwidth Memory)        │
│  • Slower: ~1.5 TB/s bandwidth      │
│  • Large: ~40 GB                    │
└─────────────────────────────────────┘
```

**Standard attention:** Constantly moves n² matrix between HBM ↔ SRAM  
**Flash Attention:** Never materializes full n² matrix, stays in SRAM

---

## 📐 Algorithm

### Tiling Strategy

```python
# Conceptual overview
Block_size = SRAM_size / (d * 3)  # Fit Q, K, V blocks

# Divide Q, K, V into blocks
Q_blocks = split(Q, block_size)
K_blocks = split(K, block_size)
V_blocks = split(V, block_size)

output = zeros(n, d)

for Q_block in Q_blocks:
    # Load Q_block into SRAM once
    for K_block, V_block in zip(K_blocks, V_blocks):
        # Compute attention for this block pair
        # Use online softmax to accumulate results
        output_block = block_attention(Q_block, K_block, V_block)
        accumulate(output, output_block)

# Result: Exact attention, O(n) memory!
```

### Online Softmax

Key technique enabling block-wise computation:

```
Standard: softmax(x) = exp(x) / Σexp(x)
          → Need all values at once

Online: Update running max and sum
        → Process in blocks!
```

---

## 💻 Implementation

### PyTorch 2.0+

```python
import torch
import torch.nn.functional as F

q = torch.randn(batch, seq_len, n_heads, head_dim).cuda()
k = torch.randn(batch, seq_len, n_heads, head_dim).cuda()
v = torch.randn(batch, seq_len, n_heads, head_dim).cuda()

# Automatically uses Flash Attention if available!
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True  # For autoregressive models
)
```

### Flash Attention Library

```python
from flash_attn import flash_attn_func

# Shape: (batch, seqlen, nheads, headdim)
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    causal=True,
    window_size=(-1, -1),  # Full attention
    return_attn_probs=False
)

# For variable-length sequences
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q,  # Cumulative sequence lengths
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal=True
)
```

---

## 📊 Performance

### Memory Complexity

| Method | Memory | Can Handle |
|--------|--------|------------|
| Standard | O(n²) | ~2K tokens |
| Flash Attention | O(n) | 100K+ tokens |
| Flash Attention 2 | O(n) | 100K+ tokens, 2x faster |

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

- **GPT-4**: Long context windows
- **Claude 2/3**: 100K-200K token context
- **Llama 2**: 4K → 32K context extension
- **MPT**: 65K token support
- **Code models**: Entire file contexts

### Applications Enabled

1. **Long Document QA**: Process full papers, books
2. **Code Understanding**: Analyze entire repositories
3. **Long-form Generation**: Write coherent long articles
4. **Conversation History**: Remember full chat context

---

## 🔄 Variants and Extensions

### Flash Attention 2 (2023)
- 2x faster than Flash 1
- Better parallelization
- Tuned for A100/H100

### Flash Attention 3 (2024)
- Support for:
  - Multi-query attention (MQA)
  - Grouped-query attention (GQA)
  - Sparse attention patterns
  - Block-diagonal masks

### PagedAttention (vLLM)
- Complementary technique
- Optimizes KV cache management
- Combined with Flash for serving

---

## 📖 Detailed Content

[→ Flash Attention Technical Details](./flash-attention.md)

---

## 📚 Resources

### Papers
- **Flash Attention** (Dao et al., NeurIPS 2022)
- **Flash Attention 2** (Dao, 2023)
- **Self-Attention Does Not Need O(n²) Memory** (Rabe & Staats, 2021)

### Code
- GitHub: `Dao-AILab/flash-attention`
- PyTorch: Built-in since 2.0
- HuggingFace: Integrated in Transformers

### Blogs
- **Making Deep Learning Go Brrrr** - Horace He
- **Flash Attention Explained** - Aman Arora

---

⬅️ [Back: Hot Topics](../) | ➡️ [Next: LoRA](../lora/)


---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
