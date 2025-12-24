<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Efficient Transformers&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# ⚡ Efficient Transformer Variants

> **Transformers with O(n) or O(n log n) attention**

<img src="./images/efficient-transformers.svg" width="100%">

---

## 📂 Topics

| File | Topic | Complexity |
|------|-------|------------|

---

## 🎯 The Problem

```
Standard Attention: O(n²)

Attention(Q,K,V) = softmax(QK^T / √d) V
                   -----------------
                   n×n matrix!

For n = 8192: 67 million scores per head!
For n = 100K: 10 billion scores! (impossible)
```

---

## 💡 Solutions

### Linformer: Project to Low Dimension

```
Standard: Q(n×d) × K^T(d×n) = n×n
Linformer: Q(n×d) × E(d×k) × K^T(k×n) = n×k (k << n)

Project K,V to lower dimension → O(nk) instead of O(n²)
```

### Performer: Kernel Approximation

```
Standard: softmax(QK^T)V
Performer: φ(Q)(φ(K)^T V)

Where φ is random feature map
Computes (φ(K)^T V) first → O(n) instead of O(n²)
```

---

## 📊 Comparison

| Method | Complexity | Quality | Long Context |
|--------|------------|---------|--------------|
| Standard | O(n²) | Best | ≤8K tokens |
| Linformer | O(n) | Good | 100K+ tokens |
| Performer | O(n) | Good | Unlimited |
| Flash Attention | O(n²) | Best | ~128K tokens |

---

## 📐 Complexity Analysis

<img src="./images/attention-complexity.svg" width="100%">

---

## 🔗 Where This Topic Is Used

| Topic | How Efficient Transformers Are Used |
|-------|-------------------------------------|
| **Long Documents** | Process full books |
| **Genomics** | Very long sequences |
| **Code Analysis** | Entire codebase context |
| **Video** | Process video frames |

---

## 📚 References & Resources

### 📄 Key Papers

| Paper | Authors | Year | Key Contribution |
|-------|---------|------|------------------|
| [Linformer](https://arxiv.org/abs/2006.04768) | Wang et al. | 2020 | O(n) via projection |
| [Performer](https://arxiv.org/abs/2009.14794) | Choromanski et al. | 2020 | FAVOR+ random features |
| [Flash Attention](https://arxiv.org/abs/2205.14135) | Dao et al. | 2022 | IO-aware attention |
| [Flash Attention 2](https://arxiv.org/abs/2307.08691) | Dao | 2023 | 2× faster flash |
| [Longformer](https://arxiv.org/abs/2004.05150) | Beltagy et al. | 2020 | Local + global attention |
| [BigBird](https://arxiv.org/abs/2007.14062) | Zaheer et al. | 2020 | Sparse attention patterns |
| [Reformer](https://arxiv.org/abs/2001.04451) | Kitaev et al. | 2020 | LSH attention |
| [Efficient Transformers Survey](https://arxiv.org/abs/2009.06732) | Tay et al. | 2020 | Comprehensive survey |
| 🇨🇳 Flash Attention详解 | [知乎](https://zhuanlan.zhihu.com/p/548036530) | - | IO感知注意力原理 |
| 🇨🇳 高效Transformer总结 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/118461491) | - | 各类线性注意力对比 |
| 🇨🇳 长文本处理 | [机器之心](https://www.jiqizhixin.com/articles/2023-11-27-7) | - | 100K+上下文技术 |

---

⬅️ [Back: Efficient Networks](../efficient-networks/)


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
