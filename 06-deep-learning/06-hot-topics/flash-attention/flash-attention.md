<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Flash%20Attention&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Flash Attention

> **IO-aware exact attention algorithm**

---

## 📐 The Problem

```
Standard attention:
Q, K, V: n × d each
S = QKᵀ: n × n (must materialize!)
A = softmax(S): n × n
O = AV: n × d

Memory: O(n²) - prohibitive for long sequences!
```

---

## 🔑 Key Insight

```
GPU memory hierarchy:
• SRAM (fast): ~20MB
• HBM (slow): ~40GB

Standard: Move n² matrix between HBM ↔ SRAM
Flash: Never materialize full n² matrix!
```

---

## 📐 Algorithm Sketch

```
Tile Q, K, V into blocks that fit in SRAM

For each block of Q:
    For each block of K, V:
        Compute block attention in SRAM
        Accumulate results (online softmax)
    
Result: Exact attention, O(n) memory!
```

---

## 💻 Code

```python
# In PyTorch 2.0+
import torch.nn.functional as F

# Automatically uses Flash Attention if available
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True  # For autoregressive models
)

# Or with flash_attn library
from flash_attn import flash_attn_func

output = flash_attn_func(
    q, k, v,  # (batch, seqlen, nheads, headdim)
    causal=True
)
```

---

## 📊 Performance

| Method | Memory | Speed |
|--------|--------|-------|
| Standard | O(n²) | Baseline |
| Flash Attention | O(n) | 2-4x faster |
| Flash Attention 2 | O(n) | 5-7x faster |

---

## 🌍 Enabled by Flash Attention

| Application | Sequence Length |
|-------------|-----------------|
| Standard BERT | 512 |
| LLaMA | 4096 |
| With Flash | 100K+ |

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
