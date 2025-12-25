<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=SVD Decomposition&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🔷 Singular Value Decomposition

> **The most versatile matrix decomposition in ML**

---

## 🎯 Visual Overview

<img src="./images/svd-visual.svg" width="100%">

*Caption: SVD decomposes any matrix A into U·Σ·Vᵀ where U and V are orthogonal and Σ contains singular values.*

---

## 📐 Mathematical Foundation

```
A = UΣVᵀ

For m×n matrix A:
• U: m×m orthogonal (left singular vectors)
• Σ: m×n diagonal (σ₁ ≥ σ₂ ≥ ... ≥ 0)
• V: n×n orthogonal (right singular vectors)

Key Properties:
• Works for ANY matrix (not just square)
• Singular values are always non-negative
• Provides best low-rank approximation
```

---

## 💻 Code Examples

```python
import numpy as np

A = np.random.randn(100, 50)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Low-rank approximation
k = 10
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

---

## 🔗 Applications

| Application | Usage |
|-------------|-------|
| **PCA** | SVD of centered data |
| **LoRA** | Low-rank weight adaptation |
| **Compression** | Image/model compression |
| **Recommenders** | Matrix factorization |

---

⬅️ [Back: Decompositions](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

