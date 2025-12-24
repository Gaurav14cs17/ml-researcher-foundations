<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Weight Sharing&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔗 Weight Sharing

> **Reducing parameters by sharing weights**

<img src="./images/weight-sharing.svg" width="100%">

---

## 📐 Mathematical Foundations

### Weight Tying
```
Instead of: y = W_out × h, embed = W_in × x
Use: W_out = W_in.T (transpose sharing)

Parameter savings: vocab_size × hidden_dim
For LLaMA-7B: 32000 × 4096 × 4 = 500 MB saved!
```

### Cross-Layer Sharing (ALBERT)
```
BERT: Each layer l has unique Wₗ
ALBERT: All layers share W₁ = W₂ = ... = Wₗ

Params: L × P → P (L times reduction)
For 12 layers: 12x parameter reduction
```

### Hash-Based Sharing
```
w_ij = shared_weights[hash(i, j) mod K]

Only K unique weights stored
Original: m × n params
Hashed: K params (K << m × n)
```

---

## 📂 Topics

| File | Topic | Key Concepts |
|------|-------|--------------|

---

## 🎯 The Core Idea

```
Original:
+-------------------------------------+
|  W₁ = [1.2, 0.8, 0.3, 0.9]         |
|  W₂ = [0.5, 1.1, 0.7, 0.2]         |
|  Parameters: 8                      |
+-------------------------------------+

With Weight Tying (W₁ = W₂):
+-------------------------------------+
|  W = [1.2, 0.8, 0.3, 0.9]          |
|  W₁ = W, W₂ = W                     |
|  Parameters: 4 (2x reduction!)      |
+-------------------------------------+
```

---

## 🌍 Famous Example: ALBERT

```
BERT:
+-- 12 transformer layers
+-- Each layer has unique weights
+-- 110M parameters

ALBERT:
+-- 12 transformer layers
+-- ALL layers share the SAME weights!
+-- 12M parameters (9x smaller!)
+-- Similar performance
```

---

## 🔗 Where This Topic Is Used

| Topic | How Weight Sharing Is Used |
|-------|---------------------------|
| **ALBERT** | Cross-layer parameter sharing |
| **Tied Embeddings** | Input/output embedding shared |
| **Transformer-XL** | Relative position shared |
| **Universal Transformers** | Shared layers |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | ALBERT | [arXiv](https://arxiv.org/abs/1909.11942) |
| 📄 | Tied Embeddings | [arXiv](https://arxiv.org/abs/1608.05859) |
| 📄 | Universal Transformer | [arXiv](https://arxiv.org/abs/1807.03819) |
| 🇨🇳 | 参数共享详解 | [知乎](https://zhuanlan.zhihu.com/p/70219212) |
| 🇨🇳 | ALBERT解析 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/125689573) |

---

⬅️ [Back: Weight Clustering](../weight-clustering/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
