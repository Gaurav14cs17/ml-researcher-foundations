# 🕸️ Sparsity and Efficient Computation

> **Using zeros and conditional computation for efficiency**

<img src="./images/sparsity-overview.svg" width="100%">

---

## 📐 Mathematical Foundations

### Sparse Matrix Computation
```
Dense: y = Wx → O(mn) operations
Sparse: y = Wx → O(nnz) operations

nnz = number of non-zeros
Speedup = mn / nnz
```

### Sparsity Patterns
```
Unstructured: Any element can be zero
Structured: Blocks, N:M (e.g., 2:4)

2:4 Sparsity (NVIDIA Ampere):
• 2 values per 4-element block
• Hardware accelerated: 2x speedup
```

### MoE Routing
```
Output y = Σᵢ gᵢ(x) · Eᵢ(x)

Where:
• gᵢ(x) = softmax(W_router · x)ᵢ  (routing weights)
• Eᵢ(x) = expert network output
• Top-k: Only k experts with highest gᵢ are computed
```

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [sparse-networks/](./sparse-networks/) | Sparse weights | Unstructured, block sparsity |
| [moe/](./moe/) | Mixture of Experts | 🔥 Conditional computation |

---

## 🎯 The Core Idea

### Sparse Weights

```
Dense matrix:
+-----------------+
| 1.2  0.5  0.8  |  All elements computed
| 0.3  0.9  0.1  |
| 0.7  0.2  0.6  |
+-----------------+

Sparse matrix (50%):
+-----------------+
| 1.2   0   0.8  |  Only non-zeros computed
|  0   0.9   0   |  → 2x theoretical speedup
| 0.7   0   0.6  |
+-----------------+
```

### Mixture of Experts

```
Input x
    |
    v
+-------------------------------------+
|           Router g(x)               |
|   "Which experts should handle x?"  |
+-------------------------------------+
    |
    +--> Expert 1 (activated) --+
    +--> Expert 2 (skipped)     |
    +--> Expert 3 (activated) --+--> Output
    +--> Expert 4 (skipped)     |
                                |
Only 2/4 experts run!           v
→ 2x effective compute     Σ weights × outputs
```

---

## 🔥 MoE: Why It Matters

```
Dense Model:
• All parameters used for every input
• 70B params → 70B FLOPs per token

MoE Model (Mixtral):
• 8 experts, top-2 routing
• 46B total params, but only ~12B active
• Better quality at same compute!

Mixtral 8x7B:
+-- Total params: 46B
+-- Active params: ~12B (per token)
+-- Equivalent quality to 70B dense model
+-- 6x more efficient inference!
```

---

## 🔗 Where This Topic Is Used

| Topic | How Sparsity Is Used |
|-------|---------------------|
| **Mixtral** | MoE architecture |
| **Switch Transformer** | Simplified MoE routing |
| **GPT-4** | Rumored to use MoE |
| **Sparse Transformers** | Sparse attention patterns |
| **Hardware Optimization** | Sparse tensor cores |
| **Pruning** | Creates sparse networks |

### Prerequisite For

```
Sparsity --> Understanding MoE (Mixtral, etc.)
        --> Efficient scaling
        --> Hardware-aware ML
        --> Sparse training
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Lottery Ticket Hypothesis | [arXiv](https://arxiv.org/abs/1803.03635) |
| 📄 | Mixtral MoE | [arXiv](https://arxiv.org/abs/2401.04088) |
| 📄 | Switch Transformer | [arXiv](https://arxiv.org/abs/2101.03961) |
| 🇨🇳 | 稀疏网络详解 | [知乎](https://zhuanlan.zhihu.com/p/84176060) |
| 🇨🇳 | MoE架构解析 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) |
| 🇨🇳 | 稀疏训练技术 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: 05-Factorization](../05-factorization/) | ➡️ [Next: 07-Efficient Architectures](../07-efficient-architectures/)

